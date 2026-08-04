from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Callable
from classifiers import (
    Dataset, 
    KNNClassifier, 
    DecisionTreeClassifier,
    WeightedKNNClassifier,
    DecisionTreePlusClassifier
)


@dataclass
class Metrics:
    accuracy: float
    precision_died: float
    recall_died: float
    f1_died: float
    tp: int
    tn: int
    fp: int
    fn: int


def remove_header(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    if not lines:
        raise ValueError("Input file is empty.")

    with open(output_path, "w", encoding="utf-8") as fout:
        for line in lines[1:]:
            fout.write(line)


def load_labeled_rows(filename: str) -> list[list[str]]:
    rows: list[list[str]] = []

    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                raise ValueError(f"Invalid labeled row at line {line_num}.")
            rows.append(parts)

    if not rows:
        raise ValueError("Labeled file is empty.")

    expected_len = len(rows[0])
    for row_num, row in enumerate(rows, start=1):
        if len(row) != expected_len:
            raise ValueError(
                f"Inconsistent row length in labeled file at row {row_num}."
            )

    return rows


def write_rows(rows: list[list[str]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(row) + "\n")


def train_test_split_rows(
    rows: list[list[str]],
    test_ratio: float = 0.2,
) -> tuple[list[list[str]], list[list[str]]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    died_rows = [row for row in rows if row[-1] == "died"]
    survived_rows = [row for row in rows if row[-1] == "survived"]

    if len(died_rows) + len(survived_rows) != len(rows):
        raise ValueError("Unknown class label found. Expected only 'died' or 'survived'.")

    died_test_size = round(len(died_rows) * test_ratio)
    survived_test_size = round(len(survived_rows) * test_ratio)

    test_rows = died_rows[:died_test_size] + survived_rows[:survived_test_size]
    train_rows = died_rows[died_test_size:] + survived_rows[survived_test_size:]

    if not train_rows or not test_rows:
        raise ValueError("Split produced an empty train or test set.")

    return train_rows, test_rows


def make_stratified_folds(
    rows: list[list[str]],
    n_folds: int = 10,
) -> list[list[list[str]]]:
    if n_folds <= 1:
        raise ValueError("n_folds must be at least 2.")

    died_rows = [row for row in rows if row[-1] == "died"]
    survived_rows = [row for row in rows if row[-1] == "survived"]

    if len(died_rows) + len(survived_rows) != len(rows):
        raise ValueError("Unknown class label found. Expected only 'died' or 'survived'.")

    folds: list[list[list[str]]] = [[] for _ in range(n_folds)]

    for i, row in enumerate(died_rows):
        folds[i % n_folds].append(row)

    for i, row in enumerate(survived_rows):
        folds[i % n_folds].append(row)

    sizes = [len(fold) for fold in folds]
    if max(sizes) - min(sizes) > 1:
        raise ValueError("Fold sizes vary by more than one.")

    return folds



def save_folds_csv(folds: list[list[list[str]]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, fold in enumerate(folds, start=1):
            f.write(f"fold{i}\n")
            for row in fold:
                f.write(",".join(row) + "\n")
            if i != len(folds):
                f.write("\n")


def load_folds_csv(filename: str) -> list[list[list[str]]]:
    folds: list[list[list[str]]] = []
    current_fold: list[list[str]] | None = None
    expected_fold_num = 1

    with open(filename, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("fold"):
                expected_name = f"fold{expected_fold_num}"
                if line != expected_name:
                    raise ValueError(
                        f"Expected fold name '{expected_name}', found '{line}'."
                    )
                current_fold = []
                folds.append(current_fold)
                expected_fold_num += 1
            else:
                if current_fold is None:
                    raise ValueError("Found data row before any fold header.")
                parts = [part.strip() for part in line.split(",")]
                current_fold.append(parts)

    if len(folds) != 10:
        raise ValueError(f"Expected 10 folds, found {len(folds)}.")

    return folds


def combine_folds(folds: list[list[list[str]]], exclude_index: int) -> Dataset:
    train_rows: list[list[str]] = []

    for i, fold in enumerate(folds):
        if i == exclude_index:
            continue
        train_rows.extend(fold)

    return Dataset.from_rows(train_rows, has_labels=True)


def fold_to_dataset(fold: list[list[str]]) -> Dataset:
    return Dataset.from_rows(fold, has_labels=True)


def compute_metrics(y_true: list[str], y_pred: list[str]) -> Metrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("Metrics require at least one example.")

    tp = tn = fp = fn = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == "died":
            if pred_label == "died":
                tp += 1
            else:
                fn += 1
        elif true_label == "survived":
            if pred_label == "died":
                fp += 1
            else:
                tn += 1
        else:
            raise ValueError(f"Unexpected true label: {true_label}")

    accuracy = (tp + tn) / len(y_true)
    precision_died = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_died = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_died = (
        2 * precision_died * recall_died / (precision_died + recall_died)
        if (precision_died + recall_died) > 0
        else 0.0
    )

    return Metrics(
        accuracy=accuracy,
        precision_died=precision_died,
        recall_died=recall_died,
        f1_died=f1_died,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )



def average_metrics(metrics_list: list[Metrics]) -> Metrics:
    if not metrics_list:
        raise ValueError("metrics_list is empty.")

    n = len(metrics_list)

    return Metrics(
        accuracy=sum(m.accuracy for m in metrics_list) / n,
        precision_died=sum(m.precision_died for m in metrics_list) / n,
        recall_died=sum(m.recall_died for m in metrics_list) / n,
        f1_died=sum(m.f1_died for m in metrics_list) / n,
        tp=sum(m.tp for m in metrics_list),
        tn=sum(m.tn for m in metrics_list),
        fp=sum(m.fp for m in metrics_list),
        fn=sum(m.fn for m in metrics_list),
    )


def cross_validate(
    classifier_factory: Callable[[], object],
    folds: list[list[list[str]]],
) -> tuple[list[Metrics], Metrics]:
    fold_metrics: list[Metrics] = []

    for test_index in range(len(folds)):
        train_data = combine_folds(folds, exclude_index=test_index)
        test_data = fold_to_dataset(folds[test_index])

        clf = classifier_factory()

        if not hasattr(clf, "fit") or not hasattr(clf, "predict"):
            raise TypeError("Classifier must provide fit(X, y) and predict(X).")

        assert train_data.y is not None
        assert test_data.y is not None

        clf.fit(train_data.X, train_data.y)
        predictions = clf.predict(test_data.X)

        metrics = compute_metrics(test_data.y.tolist(), predictions)
        fold_metrics.append(metrics)

    return fold_metrics, average_metrics(fold_metrics)


def print_metrics(metrics: Metrics, prefix: str = "") -> None:
    if prefix:
        print(prefix)
    print(f"accuracy       = {metrics.accuracy:.4f}")
    print(f"precision_died = {metrics.precision_died:.4f}")
    print(f"recall_died    = {metrics.recall_died:.4f}")
    print(f"f1_died        = {metrics.f1_died:.4f}")
    print(f"tp={metrics.tp}, tn={metrics.tn}, fp={metrics.fp}, fn={metrics.fn}")


def print_fold_metrics(metrics_list: list[Metrics]) -> None:
    for i, metrics in enumerate(metrics_list, start=1):
        print(
            f"fold{i}: "
            f"accuracy={metrics.accuracy:.4f}, "
            f"precision_died={metrics.precision_died:.4f}, "
            f"recall_died={metrics.recall_died:.4f}, "
            f"f1_died={metrics.f1_died:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation pipeline for COMP3308 A2 (NumPy version).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_strip = subparsers.add_parser("strip-header")
    p_strip.add_argument("--input", required=True)
    p_strip.add_argument("--output", required=True)

    p_split = subparsers.add_parser("split")
    p_split.add_argument("--input", required=True)
    p_split.add_argument("--train-out", required=True)
    p_split.add_argument("--test-out", required=True)
    p_split.add_argument("--test-labeled-out")
    p_split.add_argument("--test-ratio", type=float, default=0.2)

    p_make = subparsers.add_parser("make-folds")
    p_make.add_argument("--input", required=True)
    p_make.add_argument("--output", required=True)
    p_make.add_argument("--n-folds", type=int, default=10)

    p_check = subparsers.add_parser("check-folds")
    p_check.add_argument("--input", required=True)

    p_cv = subparsers.add_parser("cv")
    p_cv.add_argument("--model", choices=["dt", "knn", "knnplus", "dtplus"], required=True)
    p_cv.add_argument("--min-samples-leaf", type=int, default=3)
    p_cv.add_argument("--min-gain", type=float, default=0.01)
    p_cv.add_argument("--folds", required=True)
    p_cv.add_argument("--k", type=int, default=3)

    args = parser.parse_args()

    if args.command == "strip-header":
        remove_header(args.input, args.output)
        print(f"Wrote headerless data to {args.output}")

    elif args.command == "split":
        rows = load_labeled_rows(args.input)
        train_rows, test_rows = train_test_split_rows(rows, test_ratio=args.test_ratio)

        write_rows(train_rows, args.train_out)
        write_rows([row[:-1] for row in test_rows], args.test_out)

        if args.test_labeled_out:
            write_rows(test_rows, args.test_labeled_out)

        print(f"Wrote training data to {args.train_out}")
        print(f"Wrote unlabeled testing data to {args.test_out}")
        if args.test_labeled_out:
            print(f"Wrote labeled testing data to {args.test_labeled_out}")

        train_died = sum(1 for row in train_rows if row[-1] == "died")
        train_survived = sum(1 for row in train_rows if row[-1] == "survived")
        test_died = sum(1 for row in test_rows if row[-1] == "died")
        test_survived = sum(1 for row in test_rows if row[-1] == "survived")

        print(
            f"train: size={len(train_rows)}, died={train_died}, survived={train_survived}"
        )
        print(
            f"test:  size={len(test_rows)}, died={test_died}, survived={test_survived}"
        )

    elif args.command == "make-folds":
        rows = load_labeled_rows(args.input)
        folds = make_stratified_folds(rows, n_folds=args.n_folds)
        save_folds_csv(folds, args.output)
        print(f"Wrote {args.n_folds} folds to {args.output}")

    elif args.command == "check-folds":
        folds = load_folds_csv(args.input)
        print(f"Loaded {len(folds)} folds.")
        for i, fold in enumerate(folds, start=1):
            died = sum(1 for row in fold if row[-1] == "died")
            survived = sum(1 for row in fold if row[-1] == "survived")
            print(f"fold{i}: size={len(fold)}, died={died}, survived={survived}")

    elif args.command == "cv":
        folds = load_folds_csv(args.folds)

        if args.model == "dt":
            fold_metrics, avg_metrics = cross_validate(
                classifier_factory=lambda: DecisionTreeClassifier(),
                folds=folds,
            )

        elif args.model == "dtplus":
            fold_metrics, avg_metrics = cross_validate(
                classifier_factory=lambda: DecisionTreePlusClassifier(
                    min_samples_leaf=args.min_samples_leaf,
                ),
                folds=folds,
            )

        elif args.model == "knn":
            if args.k <= 0:
                raise ValueError("k must be a positive integer.")

            fold_metrics, avg_metrics = cross_validate(
                classifier_factory=lambda: KNNClassifier(k=args.k),
                folds=folds,
            )

        elif args.model == "knnplus":
            if args.k <= 0:
                raise ValueError("k must be a positive integer.")

            fold_metrics, avg_metrics = cross_validate(
                classifier_factory=lambda: WeightedKNNClassifier(
                    k=args.k,
                ),
                folds=folds,
            )

        print_fold_metrics(fold_metrics)
        print()
        print_metrics(avg_metrics, prefix="average:")


if __name__ == "__main__":
    main()
