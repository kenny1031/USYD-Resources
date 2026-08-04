from __future__ import annotations
import argparse
from classifiers import (
    Dataset,
    KNNClassifier,
    WeightedKNNClassifier,
    DecisionTreeClassifier,
    DecisionTreePlusClassifier,
)


def load_data(training_filename: str, testing_filename: str) -> tuple[Dataset, Dataset]:
    return (
        Dataset.from_training_file(training_filename),
        Dataset.from_testing_file(testing_filename),
    )


def classify_knn(training_filename: str, testing_filename: str, k: int) -> list[str]:
    training_data, testing_data = load_data(training_filename, testing_filename)

    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > len(training_data):
        raise ValueError("k cannot be larger than the number of training examples.")
    if training_data.n_features != testing_data.n_features:
        raise ValueError("Training and testing data must have the same number of features.")

    clf = KNNClassifier(k=k)
    assert training_data.y is not None
    clf.fit(training_data.X, training_data.y)
    return clf.predict(testing_data.X)


def classify_dt(training_filename: str, testing_filename: str) -> list[str]:
    training_data, testing_data = load_data(training_filename, testing_filename)

    if training_data.n_features != testing_data.n_features:
        raise ValueError("Training and testing data must have the same number of features.")

    clf = DecisionTreeClassifier()
    assert training_data.y is not None
    clf.fit(training_data.X, training_data.y)
    return clf.predict(testing_data.X)


def classify_knnplus(training_filename: str, testing_filename: str, k: int) -> list[str]:
    training_data, testing_data = load_data(training_filename, testing_filename)

    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > len(training_data):
        raise ValueError("k cannot be larger than the number of training examples.")
    if training_data.n_features != testing_data.n_features:
        raise ValueError("Training and testing data must have the same number of features.")

    clf = WeightedKNNClassifier(k=k, died_boost=1.3)

    assert training_data.y is not None
    clf.fit(training_data.X, training_data.y)
    
    return clf.predict(testing_data.X)
    

def classify_dtplus(training_filename: str, testing_filename: str) -> list[str]:
    training_data, testing_data = load_data(training_filename, testing_filename)

    if training_data.n_features != testing_data.n_features:
        raise ValueError("Training and testing data must have the same number of features.")

    clf = DecisionTreePlusClassifier(min_samples_split=4, min_gain=0.01)
    assert training_data.y is not None
    clf.fit(training_data.X, training_data.y)
    return clf.predict(testing_data.X)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classifiers for COMP3308 A2 (NumPy version).")
    parser.add_argument("--model", choices=["knn", "dt", "knnplus", "dtplus"], required=True,
                        help="Which classifier to run.")
    parser.add_argument("--train", required=True, help="Path to training CSV file.")
    parser.add_argument("--test", required=True, help="Path to testing CSV file.")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbours for KNN / KNN+.")
    parser.add_argument("--print-tree", action="store_true", 
                        help="Print the decision tree after training (DT only).")
    args = parser.parse_args()

    if args.model == "knn":
        predictions = classify_knn(args.train, args.test, args.k)

    elif args.model == "dt":
        training_data, testing_data = load_data(args.train, args.test)

        if training_data.n_features != testing_data.n_features:
            raise ValueError("Training and testing data must have the same number of features.")

        clf = DecisionTreeClassifier()
        assert training_data.y is not None
        clf.fit(training_data.X, training_data.y)
        predictions = clf.predict(testing_data.X)

        if args.print_tree:
            clf.print_tree()

    elif args.model == "knnplus":
        predictions = classify_knnplus(args.train, args.test, args.k)

    else:  # dtplus
        predictions = classify_dtplus(args.train, args.test)

    for pred in predictions:
        print(pred)


if __name__ == "__main__":
    main()
