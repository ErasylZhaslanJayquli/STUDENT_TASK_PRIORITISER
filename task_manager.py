"""task_manager.py — управление заданиями и связка с ML-моделью (модули 4 и 6).

Без Streamlit, поэтому всё проверяется обычными тестами (test_task_manager.py).

Задание — словарь с ключами FIELDS:
  id, title, subject, due (date), est (часы), difficulty (1–5), importance (1–5),
  status ("To Do" / "In Progress" / "Completed"), completed_on (date или None).

Цепочка «ввод → обработка → результат»:
  задание → days_left = max(due - сегодня, 0) → ml_model.predict_priority → High / Medium / Low
"""
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ml_model import predict_priority

FIELDS = ["id", "title", "subject", "due", "est", "difficulty", "importance", "status", "completed_on"]
STATUSES = ["To Do", "In Progress", "Completed"]


# --------------------------------------------------------------------------
# Создание, удаление, сортировка
# --------------------------------------------------------------------------
def new_task(task_id, title, subject, due, est, difficulty, importance, status="To Do"):
    """Создаёт задание и проверяет значения (те же границы, что у модели)."""
    title = (title or "").strip()
    if not title:
        raise ValueError("title не может быть пустым")
    if not 1 <= difficulty <= 5:
        raise ValueError("difficulty должна быть от 1 до 5")
    if not 1 <= importance <= 5:
        raise ValueError("importance должна быть от 1 до 5")
    if est <= 0:
        raise ValueError("est (часы) должно быть больше 0")
    if status not in STATUSES:
        raise ValueError(f"status должен быть одним из {STATUSES}")
    return dict(
        id=int(task_id), title=title, subject=(subject or "").strip() or "General", due=due,
        est=float(est), difficulty=int(difficulty), importance=int(importance),
        status=status, completed_on=None,
    )


def next_id(tasks):
    """Свободный id для нового задания."""
    return max((t["id"] for t in tasks), default=0) + 1


def delete_task(tasks, task_id):
    """Возвращает новый список без задания с данным id; KeyError, если такого нет."""
    if not any(t["id"] == task_id for t in tasks):
        raise KeyError(f"нет задания с id={task_id}")
    return [t for t in tasks if t["id"] != task_id]


def sort_df(df, by):
    """Сортирует таблицу заданий (колонки score, due, subject).

    "priority" — сначала самые срочные (по score модели), при равенстве — ближайший дедлайн.
    """
    if by == "priority":
        return df.sort_values(["score", "due"], ascending=[False, True])
    if by == "due":
        return df.sort_values("due")
    if by == "subject":
        return df.sort_values(["subject", "due"])
    raise ValueError("by должен быть 'priority', 'due' или 'subject'")


# --------------------------------------------------------------------------
# Сохранение и загрузка (CSV)
# --------------------------------------------------------------------------
def tasks_to_csv(tasks):
    """Список заданий -> текст CSV."""
    return pd.DataFrame(tasks, columns=FIELDS).to_csv(index=False)


def _to_date(value):
    return pd.to_datetime(value).date()


def tasks_from_csv(source):
    """CSV (путь, файловый объект) -> список заданий. ValueError при неверном формате."""
    df = pd.read_csv(source)
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"В файле нет колонок: {missing}")
    tasks = []
    for n, row in enumerate(df.to_dict("records"), start=2):  # строка 1 — заголовок
        try:
            task = new_task(
                row["id"], None if pd.isna(row["title"]) else row["title"],
                None if pd.isna(row["subject"]) else row["subject"],
                _to_date(row["due"]), row["est"], row["difficulty"], row["importance"], row["status"],
            )
            if not pd.isna(row["completed_on"]):
                task["completed_on"] = _to_date(row["completed_on"])
        except (ValueError, TypeError) as err:
            raise ValueError(f"Строка {n}: {err}") from err
        tasks.append(task)
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("В файле повторяются id заданий")
    return tasks


def save_tasks(tasks, path):
    Path(path).write_text(tasks_to_csv(tasks), encoding="utf-8")


def load_tasks(path):
    return tasks_from_csv(path)


# --------------------------------------------------------------------------
# Приоритет от ML-модели
# --------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _predict(difficulty, days_left, est, importance):
    res = predict_priority(difficulty, days_left, est, importance)
    p = res["probabilities"]
    # score — ожидаемая срочность (High=1, Medium=0,5, Low=0); нужен только для порядка внутри класса
    score = round(p.get("High", 0.0) + 0.5 * p.get("Medium", 0.0), 3)
    return res["priority"], score, max(p.values())


def predict_task(task, today=None):
    """Возвращает (приоритет, score, уверенность модели) для одного задания.

    Просроченное задание считается как days_left = 0 (самое срочное).
    """
    today = today or date.today()
    days_left = max((task["due"] - today).days, 0)
    return _predict(int(task["difficulty"]), days_left, float(task["est"]), int(task["importance"]))
