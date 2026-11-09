"""
ml_model.py — ML-модуль проекта «Student Task Prioritizer» (ICT1103, ПОН 5).

Что делает файл:
  1. Читает датасет student_task_prioritizer_dataset.csv.
  2. Делит данные на train/test (80/20, стратифицированно, random_state=42).
  3. Подбирает глубину дерева кросс-валидацией ТОЛЬКО на train-части.
  4. Сравнивает Decision Tree с другими методами (тоже только на train, CV).
  5. Один раз оценивает итоговую модель на test-части (accuracy, F1, confusion matrix).
  6. Сохраняет модель (model.joblib) и результаты (папка ml_outputs/).
  7. Даёт функции predict_priority() / predict_dataframe() для интерфейса Streamlit.

Запуск обучения и оценки:   python ml_model.py
Использование в app.py:      from ml_model import predict_priority
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # без окна: работает и на сервере, и в Colab
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_task_prioritizer_dataset.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
OUT_DIR = BASE_DIR / "ml_outputs"

FEATURES = ["difficulty", "days_left", "estimated_hours", "importance"]
TARGET = "priority"
LABELS = ["High", "Medium", "Low"]  # порядок для матрицы ошибок: от самого срочного
RANDOM_STATE = 42
TEST_SIZE = 0.2
PARAM_GRID = {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 2, 3, 5]}


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """Читает CSV и проверяет, что нужные колонки на месте и нет пропусков."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"В датасете нет колонок: {missing}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("В датасете есть пропуски — сначала очистите данные.")
    return df


# --------------------------------------------------------------------------
# Обучение и оценка
# --------------------------------------------------------------------------
def train(data_path=DATA_PATH, save=True, verbose=True):
    """Обучает Decision Tree, оценивает на test, сохраняет модель и отчёты."""
    df = load_data(data_path)
    X, y = df[FEATURES], df[TARGET]

    # 1) Разделение: test-часть не трогаем до самой финальной оценки.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 2) Подбор гиперпараметров — только по train (5-fold CV).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        PARAM_GRID,
        cv=cv,
        scoring="accuracy",
    ).fit(X_train, y_train)
    grid_table = pd.DataFrame(grid.cv_results_)[
        ["param_max_depth", "param_min_samples_leaf", "mean_test_score", "std_test_score"]
    ]
    # При равных оценках берём более простое дерево (меньше глубина, больше min_samples_leaf).
    grid_table = grid_table.sort_values(
        ["mean_test_score", "param_max_depth", "param_min_samples_leaf"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    best = grid_table.iloc[0]
    model = DecisionTreeClassifier(
        max_depth=int(best["param_max_depth"]),
        min_samples_leaf=int(best["param_min_samples_leaf"]),
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    # 3) Сравнение с альтернативными методами (CV на train, test не используется).
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    candidates = {
        "Decision Tree (выбран)": model,
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    comparison = []
    for name, est in candidates.items():
        scores = cross_val_score(est, X_train, y_train, cv=rcv, scoring="accuracy")
        comparison.append(
            {"method": name, "cv_accuracy_mean": round(scores.mean(), 3), "cv_accuracy_std": round(scores.std(), 3)}
        )
    comparison = pd.DataFrame(comparison)

    # 3b) Проверка: помогают ли категориальные признаки (subject, task_type)?
    X_cat = pd.get_dummies(df[FEATURES + ["subject", "task_type"]])
    Xc_train = X_cat.loc[X_train.index]
    cat_scores = cross_val_score(
        DecisionTreeClassifier(
            max_depth=model.max_depth, min_samples_leaf=model.min_samples_leaf, random_state=RANDOM_STATE
        ),
        Xc_train,
        y_train,
        cv=rcv,
        scoring="accuracy",
    )
    num_scores = cross_val_score(model, X_train, y_train, cv=rcv, scoring="accuracy")

    # 4) Финальная оценка на test (один раз).
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    majority = y_train.value_counts().idxmax()
    baseline_acc = accuracy_score(y_test, [majority] * len(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "best_params": {"max_depth": int(model.max_depth), "min_samples_leaf": int(model.min_samples_leaf)},
        "cv_accuracy_train_mean": round(float(best["mean_test_score"]), 3),
        "cv_accuracy_train_std": round(float(best["std_test_score"]), 3),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 3),
        "test_macro_f1": round(f1_score(y_test, y_pred, average="macro", labels=LABELS, zero_division=0), 3),
        "baseline_majority_class": str(majority),
        "baseline_test_accuracy": round(baseline_acc, 3),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "per_class": {k: {m: round(v, 3) for m, v in report[k].items()} for k in LABELS},
        "feature_importance": {f: round(float(i), 3) for f, i in zip(FEATURES, model.feature_importances_)},
        "ablation_numeric_only_cv": round(float(num_scores.mean()), 3),
        "ablation_with_categorical_cv": round(float(cat_scores.mean()), 3),
        "comparison_cv": comparison.to_dict("records"),
    }

    # 5) Сохранение результатов.
    if save:
        OUT_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        grid_table.to_csv(OUT_DIR / "grid_search_cv.csv", index=False)
        comparison.to_csv(OUT_DIR / "method_comparison_cv.csv", index=False)
        (OUT_DIR / "tree_rules.txt").write_text(export_text(model, feature_names=FEATURES), encoding="utf-8")

        examples = X_test.copy()
        examples.insert(0, "task_id", df.loc[X_test.index, "task_id"])
        examples["true_priority"] = y_test
        examples["predicted_priority"] = y_pred
        examples["correct"] = examples["true_priority"] == examples["predicted_priority"]
        examples["confidence"] = proba.max(axis=1).round(2)
        examples.to_csv(OUT_DIR / "validation_examples.csv", index=False)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion matrix (test, n={len(y_test)})")
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истинный приоритет")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 7))
        plot_tree(model, feature_names=FEATURES, class_names=list(model.classes_), filled=True, rounded=True, ax=ax, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "decision_tree.png", dpi=150)
        plt.close(fig)

    if verbose:
        _print_report(metrics, grid_table, comparison, examples if save else None)
    return model, metrics


def _print_report(metrics, grid_table, comparison, examples):
    print("=== Решающее дерево: итоги ===")
    print(f"Строк: {metrics['n_rows']} (train {metrics['n_train']} / test {metrics['n_test']})")
    print(f"Лучшие параметры: {metrics['best_params']}  (CV accuracy на train: {metrics['cv_accuracy_train_mean']})")
    print(f"Test accuracy: {metrics['test_accuracy']}  | Macro F1: {metrics['test_macro_f1']}")
    print(f"База (всегда '{metrics['baseline_majority_class']}'): {metrics['baseline_test_accuracy']}")
    print("Confusion matrix (строки — истина, столбцы — прогноз):", metrics["confusion_matrix_labels"])
    for row in metrics["confusion_matrix"]:
        print("  ", row)
    print("\nСравнение методов (повторная CV на train):")
    print(comparison.to_string(index=False))
    print(f"\nПризнаки: только числовые CV={metrics['ablation_numeric_only_cv']} "
          f"| +subject/task_type CV={metrics['ablation_with_categorical_cv']}")
    print("Важность признаков:", metrics["feature_importance"])
    if examples is not None:
        print("\nПримеры из тестовой выборки:")
        print(examples.to_string(index=False))


# --------------------------------------------------------------------------
# Использование модели в интерфейсе
# --------------------------------------------------------------------------
_model_cache = None


def load_model():
    """Загружает model.joblib; если файла нет или версия sklearn не совпала — переобучает."""
    global _model_cache
    if _model_cache is None:
        try:
            _model_cache = joblib.load(MODEL_PATH)
        except Exception:
            _model_cache, _ = train(save=True, verbose=False)
    return _model_cache


def _check_input(difficulty, days_left, estimated_hours, importance):
    if not 1 <= difficulty <= 5:
        raise ValueError("difficulty должна быть от 1 до 5")
    if not 1 <= importance <= 5:
        raise ValueError("importance должна быть от 1 до 5")
    if days_left < 0:
        raise ValueError("days_left не может быть отрицательным")
    if estimated_hours <= 0:
        raise ValueError("estimated_hours должна быть больше 0")


def predict_priority(difficulty, days_left, estimated_hours, importance):
    """Возвращает приоритет одного задания и вероятности классов.

    Пример: predict_priority(4, 1, 6, 5) -> {'priority': 'High', 'probabilities': {...}}
    """
    _check_input(difficulty, days_left, estimated_hours, importance)
    model = load_model()
    row = pd.DataFrame([[difficulty, days_left, estimated_hours, importance]], columns=FEATURES)
    proba = model.predict_proba(row)[0]
    return {
        "priority": str(model.classes_[proba.argmax()]),
        "probabilities": {str(c): round(float(p), 3) for c, p in zip(model.classes_, proba)},
    }


def predict_dataframe(df):
    """Добавляет колонку predicted_priority к таблице заданий (нужны колонки FEATURES)."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"В таблице нет колонок: {missing}")
    out = df.copy()
    out["predicted_priority"] = load_model().predict(out[FEATURES])
    return out


if __name__ == "__main__":
    train()
