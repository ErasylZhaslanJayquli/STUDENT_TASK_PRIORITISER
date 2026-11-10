"""test_model.py — проверка ML-модуля (модуль 5).

Запуск (из папки проекта):  python -m unittest -v
Часть тестов заново обучает модель (около 10–20 секунд): так проверяется, что оценка идёт на отдельной test-части.
"""
import unittest

import ml_model
from ml_model import FEATURES, TARGET, load_data, predict_dataframe, predict_priority

LABELS = {"High", "Medium", "Low"}


class TestDataset(unittest.TestCase):
    def test_dataset_is_clean_and_in_range(self):
        df = load_data()
        self.assertEqual(len(df), 100)
        self.assertFalse(df[FEATURES + [TARGET]].isna().any().any())
        self.assertEqual(set(df[TARGET]), LABELS)
        self.assertTrue(df["difficulty"].between(1, 5).all())
        self.assertTrue(df["importance"].between(1, 5).all())
        self.assertTrue((df["days_left"] >= 0).all())
        self.assertTrue((df["estimated_hours"] > 0).all())


class TestPredictions(unittest.TestCase):
    def test_three_validation_examples(self):
        """Три примера из тестовой выборки (validation_examples.csv, ML_PART.md)."""
        cases = [
            ("T073", (3, 1, 5, 5), "High"),
            ("T078", (1, 14, 2, 3), "Low"),
            ("T017", (4, 2, 7, 1), "Medium"),
        ]
        for task_id, args, expected in cases:
            with self.subTest(task=task_id):
                self.assertEqual(predict_priority(*args)["priority"], expected)

    def test_probabilities_are_valid(self):
        res = predict_priority(4, 1, 6, 5)
        self.assertEqual(set(res["probabilities"]), LABELS)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=2)
        self.assertIn(res["priority"], LABELS)

    def test_known_error_case_t058(self):
        """T058: истинный приоритет High, но модель даёт Medium (разбор в ML_PART.md)."""
        self.assertEqual(predict_priority(5, 3, 8, 5)["priority"], "Medium")

    def test_invalid_input_raises_value_error(self):
        bad = [(9, 3, 2, 3), (3, -1, 2, 3), (3, 3, 0, 3), (3, 3, 2, 0)]
        for args in bad:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    predict_priority(*args)

    def test_predict_dataframe_adds_column(self):
        df = load_data().head(10)
        out = predict_dataframe(df)
        self.assertEqual(len(out), 10)
        self.assertTrue(set(out["predicted_priority"]) <= LABELS)
        self.assertNotIn("predicted_priority", df.columns)  # исходная таблица не изменена


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.metrics = ml_model.train(save=False, verbose=False)

    def test_split_sizes(self):
        m = self.metrics
        self.assertEqual((m["n_train"], m["n_test"]), (80, 20))
        self.assertEqual(m["n_train"] + m["n_test"], m["n_rows"])

    def test_beats_majority_baseline(self):
        m = self.metrics
        self.assertGreater(m["test_accuracy"], m["baseline_test_accuracy"])
        self.assertGreaterEqual(m["test_accuracy"], 0.6)

    def test_confusion_matrix_counts_all_test_rows(self):
        m = self.metrics
        self.assertEqual(sum(sum(row) for row in m["confusion_matrix"]), m["n_test"])


if __name__ == "__main__":
    unittest.main()
