# DataTransformation
BigData projects
# Digit Recognition with SVM

A minimal Machine Learning demonstration in Python. This script trains a Support Vector Machine (SVM) classifier to recognize handwritten digits using the `scikit-learn` library.

## Features

* **Data Loading & Visualization:** Loads the `digits` dataset (8x8 pixel images) and displays them using `matplotlib`.
* **Model Training:** Trains an SVM classifier on the dataset.
* **Prediction:** Tests the model's accuracy by predicting a deliberately hidden digit.
* **Persistence:** Demonstrates saving and loading the trained model to disk using `joblib`.

## Setup

Install the required dependencies:

```bash
pip install scikit-learn matplotlib joblib

