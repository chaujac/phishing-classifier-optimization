# The Impact of Hyperparameter Optimization on the Accuracy-Latency Trade-off of Phishing Classifiers

## Overview
This repository contains the experimental machine learning pipeline for our Bachelor of Science in Computer Science thesis at the University of San Carlos. The study investigates how different hyperparameter optimization techniques influence both the predictive performance and computational efficiency of ensemble classifiers used for phishing website detection.

## Classifiers and Optimization Methods
We evaluate two tree-based ensemble learning algorithms:
* **Random Forest (Bagging)**
* **Extreme Gradient Boosting / XGBoost (Boosting)**

Each classifier is tested under three configurations across four independent datasets, producing 24 total evaluations:
1. **E1 / E4:** Default Hyperparameters
2. **E2 / E5:** GridSearchCV (Exhaustive search with 5-fold cross-validation)
3. **E3 / E6:** Optuna (Bayesian optimization using TPE with 50 trials)

## Datasets
The experiments are conducted on four publicly available phishing datasets, evaluated independently:
* UCI Phishing Websites Dataset (11,055 instances)
* Web Page Phishing Detection Dataset (11,430 instances)
* PhiUSIIL Phishing URL (Website) Dataset (235,795 instances)
* Zenodo Phishing Website Dataset (10,395 instances combined)

## Project Structure
```text
phishing-classifier-optimization/
|-- data/
|   |-- raw/          # Raw CSV datasets
|   |-- processed/    # Canonical 80/20 joblib splits
|   |-- testing/      # Dataset verification scripts
|-- models/           # Trained models (baseline, gridsearch, optuna)
|-- results/          # JSON logs, metrics, and evaluation summaries
|-- src/              # Core pipeline Python scripts
|-- requirements.txt
|-- instructions.txt
|-- README.md