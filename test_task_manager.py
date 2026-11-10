"""test_task_manager.py — проверка управления заданиями и связки с моделью (модули 4 и 6).

Запуск (из папки проекта):  python -m unittest -v
"""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import task_manager as tm

TODAY = date(2026, 11, 9)


def make(task_id=1, title="Task", due=None, est=2.0, diff=3, imp=3, status="To Do", subject="ICT"):
    return tm.new_task(task_id, title, subject, due or TODAY + timedelta(days=3), est, diff, imp, status)


class TestAddDelete(unittest.TestCase):
    def test_new_task_defaults(self):
        t = tm.new_task(1, "  Essay  ", "  ", TODAY, 2, 3, 4)
        self.assertEqual((t["title"], t["subject"], t["status"]), ("Essay", "General", "To Do"))
        self.assertIsNone(t["completed_on"])

    def test_new_task_rejects_bad_values(self):
        bad = [
            dict(title=" "), dict(diff=6), dict(diff=0), dict(imp=0), dict(imp=9),
            dict(est=0), dict(status="Done"),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    make(**kwargs)

    def test_add_gives_unique_ids(self):
        tasks = []
        for _ in range(3):
            tasks.append(make(tm.next_id(tasks)))
        self.assertEqual([t["id"] for t in tasks], [1, 2, 3])

    def test_delete_removes_only_that_task(self):
        tasks = [make(1), make(2), make(3)]
        left = tm.delete_task(tasks, 2)
        self.assertEqual([t["id"] for t in left], [1, 3])
        self.assertEqual(len(tasks), 3)  # исходный список не изменён

    def test_delete_unknown_id_raises(self):
        with self.assertRaises(KeyError):
            tm.delete_task([make(1)], 99)


class TestSort(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "title": ["a", "b", "c"],
            "score": [0.1, 1.0, 0.65],
            "due": [TODAY + timedelta(days=5), TODAY + timedelta(days=9), TODAY + timedelta(days=1)],
            "subject": ["Physics", "Calculus", "Biology"],
        })

    def test_sort_by_priority_puts_highest_score_first(self):
        self.assertEqual(list(tm.sort_df(self.df, "priority").title), ["b", "c", "a"])

    def test_sort_by_due_date(self):
        self.assertEqual(list(tm.sort_df(self.df, "due").title), ["c", "a", "b"])

    def test_sort_by_subject(self):
        self.assertEqual(list(tm.sort_df(self.df, "subject").title), ["c", "b", "a"])

    def test_unknown_sort_key_raises(self):
        with self.assertRaises(ValueError):
            tm.sort_df(self.df, "color")


class TestSaveLoad(unittest.TestCase):
    def test_csv_roundtrip_keeps_everything(self):
        done = make(2, "Quiz", status="Completed")
        done["completed_on"] = TODAY
        tasks = [make(1, "Lab report", subject="Physics"), done]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.csv"
            tm.save_tasks(tasks, path)
            self.assertEqual(tm.load_tasks(path), tasks)

    def test_empty_list_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.csv"
            tm.save_tasks([], path)
            self.assertEqual(tm.load_tasks(path), [])

    def test_missing_columns_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text("id,title\n1,x\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                tm.load_tasks(path)

    def test_bad_row_is_reported_with_line_number(self):
        text = tm.tasks_to_csv([make(1)]).replace(",3,3,To Do", ",9,3,To Do")  # difficulty = 9
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Строка 2"):
                tm.load_tasks(path)

    def test_duplicate_ids_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dup.csv"
            tm.save_tasks([make(1), make(1, "Other")], path)
            with self.assertRaises(ValueError):
                tm.load_tasks(path)


class TestPriorityBridge(unittest.TestCase):
    """Цепочка: задание -> days_left -> Decision Tree -> приоритет."""

    def test_urgent_important_task_is_high(self):
        self.assertEqual(tm.predict_task(make(due=TODAY, est=5, diff=3, imp=5), TODAY)[0], "High")

    def test_far_easy_task_is_low(self):
        self.assertEqual(tm.predict_task(make(due=TODAY + timedelta(days=14), est=2, diff=1, imp=3), TODAY)[0], "Low")

    def test_overdue_counts_as_due_today(self):
        late = make(due=TODAY - timedelta(days=3), est=5, diff=3, imp=5)
        today = make(due=TODAY, est=5, diff=3, imp=5)
        self.assertEqual(tm.predict_task(late, TODAY), tm.predict_task(today, TODAY))

    def test_score_and_confidence_in_range(self):
        label, score, conf = tm.predict_task(make(), TODAY)
        self.assertIn(label, {"High", "Medium", "Low"})
        self.assertTrue(0 <= score <= 1 and 0 < conf <= 1)


if __name__ == "__main__":
    unittest.main()
