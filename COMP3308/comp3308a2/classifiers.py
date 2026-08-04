from __future__ import annotations
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any
import numpy as np

# ==========================
# Dataset
# ==========================
class Dataset:
    X: np.ndarray
    y: np.ndarray | None

    def __init__(self, X: np.ndarray, y: np.ndarray | None=None):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y is not None and y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return int(self.X.shape[0])
    
    @property
    def n_features(self) -> int:
        if self.X.ndim != 2:
            return 0
        return int(self.X.shape[1])

    @property
    def has_labels(self) -> bool:
        return self.y is not None

    @classmethod
    def from_training_file(cls, filename: str) -> Dataset:
        rows: list[list[str]] = []
        labels: list[str] = []

        with open(filename, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid training row at line {line_num}: expected at least 2 columns."
                    )

                rows.append(parts[:-1])
                labels.append(parts[-1])

        if not rows:
            raise ValueError("Training file is empty.")

        X = np.array(rows, dtype=object)
        y = np.array(labels, dtype=object)

        if X.ndim != 2:
            raise ValueError("Training data must be a 2D table.")

        return cls(X, y)

    @classmethod
    def from_testing_file(cls, filename: str) -> Dataset:
        rows: list[list[str]] = []

        with open(filename, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 1:
                    raise ValueError(
                        f"Invalid testing row at line {line_num}: expected at least 1 column."
                    )

                rows.append(parts)

        if not rows:
            raise ValueError("Testing file is empty.")

        X = np.array(rows, dtype=object)

        if X.ndim != 2:
            raise ValueError("Testing data must be a 2D table.")

        return cls(X, None)

    @classmethod
    def from_rows(cls, rows: list[list[str]], has_labels: bool = True) -> Dataset:
        if not rows:
            raise ValueError("Rows are empty.")

        arr = np.array(rows, dtype=object)
        if arr.ndim != 2:
            raise ValueError("Rows must form a 2D table.")

        if has_labels:
            if arr.shape[1] < 2:
                raise ValueError("Labeled rows must contain at least one feature and one label.")
            X = arr[:, :-1]
            y = arr[:, -1]
            return cls(X, y)

        return cls(arr, None)

    def rows_with_labels(self) -> list[list[str]]:
        if self.y is None:
            raise ValueError("This dataset has no labels.")
        return [list(row) + [str(label)] for row, label in zip(self.X, self.y)]

    def subset(self, indices: list[int]) -> Dataset:
        X_sub = self.X[indices]
        if self.y is None:
            return Dataset(X_sub, None)
        y_sub = self.y[indices]
        return Dataset(X_sub, y_sub)

# ================================
# Abstract Parent Classifier class
# ================================
class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseClassifier:
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> list[str]:
        raise NotImplementedError
    
    @abstractmethod
    def predict_one(self, x: np.ndarray) -> str:
        raise NotImplementedError
    
    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must be non-empty.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

    def _check_X(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

    def _check_is_fitted(self) -> None:
        raise NotImplementedError

# ============================
# KNeighbours Classifier class
# ============================
class KNNClassifier(BaseClassifier):
    def __init__(self, k: int=3) -> None:
        self.k = k
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
    
    def _check_is_fitted(self) -> None:
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("KNNClassifier has not been fitted yet.")
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)

        self._check_X_y(X, y)

        if self.k <= 0:
            raise ValueError("k must be a positive integer")
        if self.k > X.shape[0]:
            raise ValueError("k cannot be larger than the number of training examples")

        self.X_train_ = X
        self.y_train_ = y
        
        return self
    
    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        assert self.X_train_ is not None

        x = np.array(x, dtype=object)
        if x.ndim != 1:
            raise ValueError("A single test example must be a 1D array.")
        if x.shape[0] != self.X_train_.shape[1]:
            raise ValueError("Test example has a different number of features")

        mismatches = np.sum(self.X_train_ != x, axis=1)
        return np.sqrt(mismatches.astype(float))
        
    def _get_k_nearest(self, x: np.ndarray) -> list[tuple[float, int, str]]:
        self._check_is_fitted()
        assert self.y_train_ is not None

        distances = self._compute_distances(x)
        indexed: list[tuple[float, int, str]] = [
            (int(distances[i]), i, str(self.y_train_[i]))
            for i in range(len(distances))
        ]

        # tie-breaking
        # smaller distance first
        # if equal distance, smaller training row index first
        indexed.sort(key=lambda item: (item[0], item[1]))
        return indexed[: self.k]
    
    def _vote(self, neighbours: list[tuple[float, int, str]]) -> str:
        died_count = sum(1 for _, _, label in neighbours if label == "died")
        survived_count = sum(1 for _, _, label in neighbours if label == "survived")

        # tie-breaking: choose died
        return "died" if died_count >= survived_count else "survived"
    
    def predict_one(self, x: np.ndarray) -> str:
        neighbours = self._get_k_nearest(x)
        return self._vote(neighbours)

    def predict(self, X: np.ndarray) -> list[str]:
        self._check_is_fitted()

        X = np.array(X, dtype=object)
        self._check_X(X)

        assert self.X_train_ is not None
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError("Testing data must have the same number of features as training data.")

        return [self.predict_one(row) for row in X]

# ==============================
# KNN+
# ==============================
class WeightedKNNClassifier(KNNClassifier):
    def __init__(self, k: int = 7) -> None:
        super().__init__(k)
        self.feature_weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedKNNClassifier":
        super().fit(X, y)

        assert self.X_train_ is not None
        assert self.y_train_ is not None

        n_features = self.X_train_.shape[1]
        weights = []

        for feature_index in range(n_features):
            score = self._feature_importance(feature_index)
            weights.append(score)

        weights_array = np.array(weights, dtype=float)

        # avoid all-zero weights
        if np.sum(weights_array) == 0:
            weights_array = np.ones(n_features)

        # normalise weights so average weight is about 1
        weights_array = weights_array / np.mean(weights_array)

        self.feature_weights_ = weights_array
        return self

    def _feature_importance(self, feature_index: int) -> float:
        assert self.X_train_ is not None
        assert self.y_train_ is not None

        values = np.unique(self.X_train_[:, feature_index])
        total = len(self.y_train_)

        score = 0.0

        for value in values:
            mask = self.X_train_[:, feature_index] == value
            subset_y = self.y_train_[mask]

            died_count = np.sum(subset_y == "died")
            survived_count = np.sum(subset_y == "survived")
            subset_size = len(subset_y)

            if subset_size == 0:
                continue

            died_rate = died_count / subset_size
            survived_rate = survived_count / subset_size

            # bigger difference = more useful feature
            score += (subset_size / total) * abs(died_rate - survived_rate)

        return float(score)

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()

        assert self.X_train_ is not None
        assert self.feature_weights_ is not None

        x = np.array(x, dtype=object)

        if x.ndim != 1:
            raise ValueError("A single test example must be a 1D array.")

        if x.shape[0] != self.X_train_.shape[1]:
            raise ValueError("Test example has a different number of features.")

        differences = self.X_train_ != x

        # important features contribute more to distance
        weighted_differences = differences * self.feature_weights_

        return np.sum(weighted_differences, axis=1)

    def _weighted_vote(self, neighbours: list[tuple[float, int, str]]) -> str:
        died_score = 0.0
        survived_score = 0.0

        for dist, index, label in neighbours:
            weight = 1.0 / (dist + 1)

            if label == "died":
                died_score += weight
            elif label == "survived":
                survived_score += weight

        return "died" if died_score >= survived_score else "survived"

    def predict_one(self, x: np.ndarray) -> str:
        neighbours = self._get_k_nearest(x)
        return self._weighted_vote(neighbours)


# ==============================
# Decision Tree Classifier 
# ==============================
class DecisionTreeNode:
    def __init__(
        self, 
        attribute_index: int | None=None, 
        label: str | None=None
    ) -> None:
        self.attribute_index = attribute_index
        self.label = label
        self.children: dict[str, DecisionTreeNode] = {}
        self.majority_label: str | None = None
    
    def is_leaf(self) -> bool:
        return self.label is not None


class DecisionTreeClassifier(BaseClassifier):
    def __init__(self):
        self.root_: DecisionTreeNode | None = None
        self.n_features_in_: int | None = None

    def _check_is_fitted(self) -> None:
        if self.root_ is None or self.n_features_in_ is None:
            raise ValueError("DecisionTreeClassifier has not been fitted yet.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)

        self._check_X_y(X, y)

        self.n_features_in_ = int(X.shape[1])
        attributes = list(range(self.n_features_in_))
        self.root_ = self._build_tree(X, y, attributes)
        return self

    def predict(self, X: np.ndarray) -> list[str]:
        self._check_is_fitted()

        X = np.array(X, dtype=object)
        self._check_X(X)

        assert self.n_features_in_ is not None
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Testing data must have the same number of features as training data.")

        return [self.predict_one(row) for row in X]

    def predict_one(self, x: np.ndarray) -> str:
        self._check_is_fitted()
        assert self.root_ is not None
        assert self.n_features_in_ is not None

        x = np.array(x, dtype=object)
        if x.ndim != 1:
            raise ValueError("A single test example must be a 1D array.")
        if x.shape[0] != self.n_features_in_:
            raise ValueError("Test example has a different number of features.")

        node = self.root_

        while not node.is_leaf():
            assert node.attribute_index is not None

            value = str(x[node.attribute_index])

            if value not in node.children:
                assert node.majority_label is not None
                return node.majority_label

            node = node.children[value]

        assert node.label is not None
        return node.label

    def _entropy(self, labels: np.ndarray) -> float:
        if labels.size == 0:
            raise ValueError("Cannot compute entropy of an empty label array.")

        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()

        entropy = 0.0
        for p in probabilities:
            entropy -= float(p) * math.log2(float(p))

        return entropy

    def _majority_label(self, labels: np.ndarray) -> str:
        died_count = int(np.sum(labels == "died"))
        survived_count = int(np.sum(labels == "survived"))

        # tie-breaking: choose died
        return "died" if died_count >= survived_count else "survived"

    def _information_gain(self, X: np.ndarray, y: np.ndarray, attr_index: int) -> float:
        base_entropy = self._entropy(y)
        values = np.unique(X[:, attr_index])

        remainder = 0.0
        total = int(y.shape[0])

        for value in values:
            mask = X[:, attr_index] == value
            subset_y = y[mask]
            weight = subset_y.shape[0] / total
            remainder += float(weight) * self._entropy(subset_y)

        return base_entropy - remainder

    def _partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attr_index: int,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        partitions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        values = np.unique(X[:, attr_index])

        for value in values:
            value_str = str(value)
            mask = X[:, attr_index] == value
            partitions[value_str] = (X[mask], y[mask])

        return partitions

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attributes: list[int],
    ) -> DecisionTreeNode:
        if y.size == 0:
            raise ValueError("Cannot build a tree from an empty dataset.")

        # stopping condition 1: all labels identical
        if np.all(y == y[0]):
            return DecisionTreeNode(label=str(y[0]))

        # stopping condition 2: no attributes left
        if not attributes:
            return DecisionTreeNode(label=self._majority_label(y))

        best_attr = max(attributes, key=lambda a: self._information_gain(X, y, a))

        node = DecisionTreeNode(attribute_index=best_attr)
        node.majority_label = self._majority_label(y)

        partitions = self._partition(X, y, best_attr)
        remaining_attributes = [a for a in attributes if a != best_attr]

        for attr_value, (subset_X, subset_y) in partitions.items():
            child = self._build_tree(subset_X, subset_y, remaining_attributes)
            node.children[attr_value] = child

        return node

    def print_tree(self) -> None:
        self._check_is_fitted()
        assert self.root_ is not None
        self._print_tree(self.root_, depth=0)

    def _print_tree(self, node: DecisionTreeNode, depth: int) -> None:
        indent = "  " * depth

        if node.is_leaf():
            assert node.label is not None
            print(f"{indent}Leaf: {node.label}")
            return

        assert node.attribute_index is not None
        print(f"{indent}Attribute[{node.attribute_index}]")

        for value, child in node.children.items():
            print(f"{indent}  = {value}")
            self._print_tree(child, depth + 1)


# ==========================
# Decision Tree Plus
# ==========================
class DecisionTreePlusClassifier(DecisionTreeClassifier):
    def __init__(self, min_samples_leaf: int = 3) -> None:
        super().__init__()
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreePlusClassifier:
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)

        self._check_X_y(X, y)

        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        self.n_features_in_ = int(X.shape[1])
        attributes = list(range(self.n_features_in_))
        self.root_ = self._build_tree(X, y, attributes)
        return self

    def _is_valid_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attr_index: int,
    ) -> bool:
        partitions = self._partition(X, y, attr_index)

        for subset_X, _ in partitions.values():
            if subset_X.shape[0] < self.min_samples_leaf:
                return False

        return True

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attributes: list[int],
    ) -> DecisionTreeNode:
        if y.size == 0:
            raise ValueError("Cannot build a tree from an empty dataset.")

        # stopping condition 1: all labels identical
        if np.all(y == y[0]):
            return DecisionTreeNode(label=str(y[0]))

        # stopping condition 2: no attributes left
        if not attributes:
            return DecisionTreeNode(label=self._majority_label(y))

        # only keep attributes whose split does not create tiny child nodes
        valid_attributes = [
            attr for attr in attributes if self._is_valid_split(X, y, attr)
        ]

        # stopping condition 3: no valid split remains
        if not valid_attributes:
            return DecisionTreeNode(label=self._majority_label(y))

        # choose best valid attribute by information gain
        best_attr = max(valid_attributes, key=lambda a: self._information_gain(X, y, a))

        node = DecisionTreeNode(attribute_index=best_attr)
        node.majority_label = self._majority_label(y)

        partitions = self._partition(X, y, best_attr)
        remaining_attributes = [a for a in attributes if a != best_attr]

        for attr_value, (subset_X, subset_y) in partitions.items():
            child = self._build_tree(subset_X, subset_y, remaining_attributes)
            node.children[attr_value] = child

        return node
