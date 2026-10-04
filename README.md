# The Impact of Hyperparameter Optimization on the Accuracy-Latency Trade-off of Phishing Classifiers

## Overview
This repository contains the experimental machine learning pipeline for our Bachelor of Science in Computer Science thesis at the University of San Carlos. The study investigates how different hyperparameter optimization techniques influence both the predictive performance and computational efficiency of ensemble classifiers used for phishing website detection.

## Classifiers and Optimization Methods
We evaluate two tree-based ensemble learning algorithms:
* **Random Forest (Bagging)**
* **Extreme Gradient Boosting / XGBoost (Boosting)**

Each classifier is tested under three configurations across four independent datasets, producing 24 total evaluations:
1. Default Hyperparameters
2. GridSearchCV (Exhaustive search with 5-fold cross-validation)
3. Optuna (Bayesian optimization using TPE with 50 trials)

## Datasets
The experiments are conducted on four publicly available phishing datasets, evaluated independently:
* UCI Phishing Websites Dataset (11,055 instances)
* Web Page Phishing Detection Dataset (11,430 instances)
* PhiUSIIL Phishing URL (Website) Dataset (235,795 instances)
* Zenodo Phishing Website Dataset (10,395 instances)

## Evaluation Metrics
To assess the accuracy-latency trade-off, models are evaluated on:
* **Predictive Performance:** Accuracy, Precision, Recall, and F1-score (with $0.50$ decision threshold).
* **Computational Efficiency:** Single-instance inference latency (measured over 1,000 iterations post-warm-up) and peak memory usage.
* **Optimization Cost:** Total wall-clock time required for the hyperparameter search process.

## Setup and Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/phishing-classifier-optimization.git](https://github.com/yourusername/phishing-classifier-optimization.git)
