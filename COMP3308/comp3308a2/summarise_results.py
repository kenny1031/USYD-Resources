from __future__ import annotations
import csv
import io
import json
import os
from contextlib import redirect_stdout
from classifiers import (
    Dataset,
    KNNClassifier,
    WeightedKNNClassifier,
    DecisionTreeClassifier,
    DecisionTreePlusClassifier,
)
from evaluate import load_folds_csv, cross_validate


DATA_FILE = "data/heart.csv"
FOLDS_FILE = "data/heart-folds.csv"
OUTPUT_DIR = "results"


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def dataset_summary() -> dict:
    dataset = Dataset.from_training_file(DATA_FILE)
    assert dataset.y is not None

    died_count = int((dataset.y == "died").sum())
    survived_count = int((dataset.y == "survived").sum())

    return {
        "n_examples": len(dataset),
        "n_features": dataset.n_features,
        "classes": ["died", "survived"],
        "class_counts": {
            "died": died_count,
            "survived": survived_count,
        },
        "data_type": "nominal / discretised",
    }


def folds_summary() -> list[dict]:
    folds = load_folds_csv(FOLDS_FILE)
    summary = []

    for i, fold in enumerate(folds, start=1):
        died = sum(1 for row in fold if row[-1] == "died")
        survived = sum(1 for row in fold if row[-1] == "survived")
        summary.append(
            {
                "fold": i,
                "size": len(fold),
                "died": died,
                "survived": survived,
            }
        )

    return summary


def metrics_to_dict(model_name: str, avg) -> dict:
    return {
        "model": model_name,
        "accuracy": round(avg.accuracy, 4),
        "precision_died": round(avg.precision_died, 4),
        "recall_died": round(avg.recall_died, 4),
        "f1_died": round(avg.f1_died, 4),
        "tp": avg.tp,
        "tn": avg.tn,
        "fp": avg.fp,
        "fn": avg.fn,
    }


def run_model_cv() -> list[dict]:
    folds = load_folds_csv(FOLDS_FILE)

    experiments = [
        ("MyKNN", lambda: KNNClassifier(k=3)),
        ("MyKNN+", lambda: WeightedKNNClassifier(k=7)),
        ("MyDT", lambda: DecisionTreeClassifier()),
        ("MyDT+", lambda: DecisionTreePlusClassifier(min_samples_leaf=5)),
    ]

    results = []

    for model_name, factory in experiments:
        _, avg = cross_validate(factory, folds)
        results.append(metrics_to_dict(model_name, avg))

    return results


def run_knn_sweep(k_values: list[int]) -> list[dict]:
    folds = load_folds_csv(FOLDS_FILE)
    rows = []

    for k in k_values:
        _, avg = cross_validate(lambda: KNNClassifier(k=k), folds)
        rows.append(
            {
                "k": k,
                "accuracy": round(avg.accuracy, 4),
                "precision_died": round(avg.precision_died, 4),
                "recall_died": round(avg.recall_died, 4),
                "f1_died": round(avg.f1_died, 4),
                "tp": avg.tp,
                "tn": avg.tn,
                "fp": avg.fp,
                "fn": avg.fn,
            }
        )

    return rows


def run_knnplus_sweep(k_values: list[int]) -> list[dict]:
    folds = load_folds_csv(FOLDS_FILE)
    rows = []

    for k in k_values:
        _, avg = cross_validate(lambda: WeightedKNNClassifier(k=k), folds)
        rows.append(
            {
                "k": k,
                "accuracy": round(avg.accuracy, 4),
                "precision_died": round(avg.precision_died, 4),
                "recall_died": round(avg.recall_died, 4),
                "f1_died": round(avg.f1_died, 4),
                "tp": avg.tp,
                "tn": avg.tn,
                "fp": avg.fp,
                "fn": avg.fn,
            }
        )

    return rows


def run_dtplus_sweep(min_leaf_values: list[int]) -> list[dict]:
    folds = load_folds_csv(FOLDS_FILE)
    rows = []

    for min_leaf in min_leaf_values:
        _, avg = cross_validate(
            lambda: DecisionTreePlusClassifier(min_samples_leaf=min_leaf),
            folds,
        )
        rows.append(
            {
                "min_samples_leaf": min_leaf,
                "accuracy": round(avg.accuracy, 4),
                "precision_died": round(avg.precision_died, 4),
                "recall_died": round(avg.recall_died, 4),
                "f1_died": round(avg.f1_died, 4),
                "tp": avg.tp,
                "tn": avg.tn,
                "fp": avg.fp,
                "fn": avg.fn,
            }
        )

    return rows


def capture_tree_text() -> str:
    dataset = Dataset.from_training_file(DATA_FILE)
    assert dataset.y is not None

    dt = DecisionTreeClassifier()
    dt.fit(dataset.X, dataset.y)

    dtplus = DecisionTreePlusClassifier(min_samples_leaf=5)
    dtplus.fit(dataset.X, dataset.y)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("=== MyDT (whole dataset) ===")
        dt.print_tree()
        print()
        print("=== MyDT+ (whole dataset) ===")
        dtplus.print_tree()

    return buffer.getvalue()


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    ensure_output_dir()

    dataset_info = dataset_summary()
    fold_info = folds_summary()
    model_info = run_model_cv()
    knn_sweep = run_knn_sweep([1, 3, 5, 7, 9])
    knnplus_sweep = run_knnplus_sweep([1, 3, 5, 7, 9])
    dtplus_sweep = run_dtplus_sweep([2, 3, 4, 5])
    tree_text = capture_tree_text()

    write_json(os.path.join(OUTPUT_DIR, "dataset_summary.json"), dataset_info)
    write_csv(os.path.join(OUTPUT_DIR, "fold_summary.csv"), fold_info)
    write_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"), model_info)
    write_json(os.path.join(OUTPUT_DIR, "model_summary.json"), model_info)
    write_csv(os.path.join(OUTPUT_DIR, "knn_sweep.csv"), knn_sweep)
    write_csv(os.path.join(OUTPUT_DIR, "knnplus_sweep.csv"), knnplus_sweep)
    write_csv(os.path.join(OUTPUT_DIR, "dtplus_sweep.csv"), dtplus_sweep)
    write_text(os.path.join(OUTPUT_DIR, "dt_trees.txt"), tree_text)

    print("Wrote:")
    print(f"  {OUTPUT_DIR}/dataset_summary.json")
    print(f"  {OUTPUT_DIR}/fold_summary.csv")
    print(f"  {OUTPUT_DIR}/model_summary.csv")
    print(f"  {OUTPUT_DIR}/model_summary.json")
    print(f"  {OUTPUT_DIR}/knn_sweep.csv")
    print(f"  {OUTPUT_DIR}/knnplus_sweep.csv")
    print(f"  {OUTPUT_DIR}/dtplus_sweep.csv")
    print(f"  {OUTPUT_DIR}/dt_trees.txt")


if __name__ == "__main__":
    main()
