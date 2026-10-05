# MLOps Assignment 1 – Heart Disease Prediction

## 1. Project Overview

This project is part of the MLOps assignment and focuses on data acquisition,
data preprocessing, and exploratory data analysis (EDA) using the UCI Heart
Disease dataset.

The objective of this task is to:

- Acquire the dataset from a reliable source
- Understand the structure and quality of the data
- Handle missing values
- Check for duplicate records
- Analyze numerical and categorical feature distributions
- Encode categorical features
- Create a machine-learning-ready dataset
- Perform exploratory data analysis
- Analyze feature relationships and class balance
- Prepare the dataset for subsequent machine learning and MLOps tasks

---

## 2. Dataset

### Dataset Name

UCI Heart Disease Dataset

### Dataset Source

The dataset is obtained from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/45/heart+disease

### Dataset Used

For this assignment, the **processed Cleveland dataset** was used:

`processed.cleveland.data`

The dataset contains **303 patient records** and **14 original attributes**.

The attribute definitions were verified using:

`heart-disease.names`

---

## 3. Dataset Acquisition

The dataset can be downloaded using the provided Python script:

```text
src/download_data.py