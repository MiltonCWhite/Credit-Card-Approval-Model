# Credit-Card-Approval-Model
This repository contains a machine learning project focused on predicting the approval of credit card applications. The dataset used in this project consists of various features related to applicants, and the goal is to build models that can accurately predict whether an application will be approved or not.

Table of Contents
Project Overview
Dataset
Preprocessing
Modeling
Results
Dependencies
Usage
Contributing
License

Project Overview
The Credit-Card-Approval-Model project involves the following steps:
Data Exploration and Preprocessing: Understanding the dataset, handling missing values, and preparing the data for modeling.
Feature Engineering: Converting categorical variables into numerical ones and scaling features.
Model Training: Building and training different machine learning models to predict credit card approval.
Model Evaluation: Evaluating the models' performance using metrics such as accuracy, precision, recall, and F1-score.


Dataset
The dataset contains 690 records with 14 attributes each, including:
Gender: Binary (0, 1)
Age: Numeric
Debt: Numeric
Married: Binary (0, 1)
BankCustomer: Binary (0, 1)
Industry: Categorical
Ethnicity: Categorical
YearsEmployed: Numeric
PriorDefault: Binary (0, 1)
Employed: Binary (0, 1)
CreditScore: Numeric
Citizen: Categorical
Income: Numeric
Approved: Target variable (0: Not Approved, 1: Approved)

Preprocessing
Several preprocessing steps were applied to the dataset:
Dropped irrelevant columns (DriversLicense, ZipCode).
Converted categorical variables (Industry, Ethnicity, Citizen) into numerical form.
Split the data into training and testing sets.
Applied feature scaling using StandardScaler and MinMaxScaler.

Modeling
Multiple models were built to predict credit card approval:
K-Nearest Neighbors (KNN)
Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)

Each model was trained on the training set and evaluated on the test set. The performance of the models was compared based on their accuracy and other metrics.

Results
The models achieved the following accuracy on the test set:
K-Nearest Neighbors: 78.61%
Logistic Regression: 84.39%
Random Forest Classifier: 87.28%
Support Vector Machine: 84.39%
Random Forest Classifier performed the best in terms of accuracy, with a precision of 0.88 and recall of 0.84.

A feature selection technique, Recursive Feature Elimination (RFE), was also applied to identify the most important features. The top features identified were:
Industry
YearsEmployed
PriorDefault
Employed
CreditScore
Citizen

Dependencies
This project requires the following Python libraries:
numpy
pandas
plotly
seaborn
matplotlib
scikit-learn

You can install the required dependencies using the following command:

---bash
pip install -r requirements.txt

Usage
To use this project, follow these steps:
Clone the repository:
---bash
git clone https://github.com/yourusername/Credit-Card-Approval-Model.git

Navigate to the project directory:
---bash
cd Credit-Card-Approval-Model

Run the notebook or Python scripts to train and evaluate models.

Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to create a pull request or open an issue.

This README provides an overview of the project, including its goals, methods, and results. For a more detailed analysis, refer to the code and accompanying notebooks in this repository.







