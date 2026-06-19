# Data Transformation
BigData projects I did as assignments for my university classes
# 1. Digit Recognition with SVM

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
```
## Usage

1. Open the script and update the model_path variable to a valid local directory (for clf.joblib to save in the current folder).
2. Run the script from your terminal:

```bash
python digitrecognition.py
```

# 2. Flu Prediction with Decision Trees 

A minimal Machine Learning demonstration using Python. This script trains a Decision Tree Classifier using the `scikit-learn` library to predict the presence of the flu based on various symptoms (chills, runny nose, fever, headache).

## Features

* **Data Cleaning & Preprocessing:** Standardizes text data by stripping whitespace and converting strings to lowercase to prevent mapping errors.
* **Categorical Encoding:** Converts text labels (e.g., 'tak', 'nie', 'sredni', 'duzy') into numerical formats required by the model.
* **Model Training:** Trains a `DecisionTreeClassifier` on the cleaned training dataset.
* **Prediction & Evaluation:** Predicts outcomes for both the test and training datasets and calculates the model's accuracy.

## Setup

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn
```
## Usage

Run the script from the terminal:

```bash
python decisiontree.py
```

## Expected Output
```
Prediction for test dataset:
[1]

Prediction for training dataset:
[0 1 1 1 0 1 0 1]

Accuracy on training dataset: 1.0
```
