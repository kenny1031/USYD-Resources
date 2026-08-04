# COMP3308 Assignment 2

This repository contains our code and experiment files for COMP3308 Assignment 2.

## Files

- `program.py`  
  Main entry point for running the classifiers required by the assignment.

- `classifiers.py`  
  Contains the implementations of:
  - KNN
  - KNN+
  - Decision Tree
  - Decision Tree+

- `evaluate.py`  
  Utilities for:
  - train/test splitting
  - generating stratified 10-fold cross validation folds
  - evaluating classifiers

- `data/`  
  Stores the dataset files:
  - `heart.csv`: processed dataset without header
  - `heart-folds.csv`: stratified 10-fold split
  - `train.csv`, `test.csv`, `test_with_labels.csv`: local testing files

- `results/`  
  Stores experiment outputs and summaries.

- `summarise_results.py`  
  Helper script for summarising evaluation results.

## How to run

### Run KNN
```bash
python program.py --model knn \
  --train data/train.csv \
  --test data/test.csv \
  --k 3
```

### Run Decision Tree
```bash
python program.py --model dt \
  --train data/train.csv \
  --test data/test.csv
```

### Print the Decision Tree
```bash
python program.py --model dt \
  --train data/train.csv \
  --test data/test.csv \
  --print-tree
```

## Evaluation
### Generate stratified folds
```bash
python evaluate.py make-folds \
  --input data/heart.csv \
  --output data/heart-folds.csv \
```

### Check folds
```bash
python evaluate.py check-folds --input data/heart-folds.csv
```

### Cross-validation for KNN
```bash
python evaluate.py cv --model knn \
  --folds data/heart-folds.csv \
  --k 3
```

### Cross-validation for Decision Tree
```bash
python evaluate.py cv --model dt --folds data/heart-folds.csv
```

### Cross-validation KNN+
```bash
python evaluate.py cv --model knnplus \
  --folds data/heart-folds.csv \
  --k 3
```

### Cross-validation DT+
```bash
python evaluate.py cv --model dtplus \
  --folds data/heart-folds.csv \
  --min-samples-leaf 5
```

### Output results artifacts
```bash
python summarise_results.py
```
