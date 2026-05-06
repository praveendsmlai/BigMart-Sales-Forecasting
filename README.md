# BigMart Sales Prediction

An end-to-end Machine Learning and MLOps project for predicting BigMart product sales using feature engineering, preprocessing pipelines, model training, and hyperparameter tuning.

---

# Project Overview

This project predicts the `Item_Outlet_Sales` of products sold across BigMart outlets.

The pipeline includes:

* Data Ingestion
* Data Validation
* Feature Engineering
* Data Transformation
* Model Training
* Hyperparameter Tuning
* Model Evaluation
* Artifact Management
* Logging & Exception Handling

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* YAML
* Git & GitHub

---

# Problem Statement

BigMart wants to predict future sales for products across multiple outlets based on:

* Product attributes
* Outlet details
* Pricing
* Visibility
* Fat content
* Item category
* Outlet characteristics

This project builds a regression pipeline to solve this business problem.

---

# Dataset Features

## Numerical Features

* Item_Weight
* Item_Visibility
* Item_MRP
* Outlet_Age

## Categorical Features

* Item_Fat_Content
* Item_Type
* Outlet_Identifier
* Outlet_Size
* Outlet_Location_Type
* Outlet_Type
* Item_Category

## Target Feature

* Item_Outlet_Sales

---

# Feature Engineering

The following custom features are created:

## Item Category

Extracted from `Item_Identifier`

Examples:

* FD → Food
* DR → Drinks
* NC → Non-consumable

## Outlet Age

```python
Outlet_Age = 2026 - Outlet_Establishment_Year
```

## Visibility Handling

Zero visibility values are replaced with NaN.

---

# Machine Learning Pipeline

The preprocessing pipeline includes:

## Numerical Pipeline

* Missing value imputation
* Standard scaling

## Categorical Pipeline

* Missing value imputation
* One-hot encoding

## Feature Engineering Pipeline

Implemented using:

```python
FunctionTransformer
```

---

# Models Used

* Linear Regression
* KNeighbors Regressor
* Decision Tree Regressor
* Random Forest Regressor

---

# Hyperparameter Tuning

Hyperparameter tuning is performed using:

```python
GridSearchCV
```

---

# Evaluation Metrics

The following metrics are used:

* RMSE
* MAE
* R² Score

---

# Project Structure

```text
Bigmart_Sales_Prediction/
│
├── artifacts/
├── final_model/
├── notebook/
├── logs/
│
├── Bigmart_Sales_Prediction/
│
```
