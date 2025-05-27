# Saturation Vapour Pressure Prediction

## 🧪 Project Overview
This project was developed as part of the Introduction to Machine Learning course at the University of Helsinki. The aim was to build a predictive model for estimating the saturation vapour pressure (pSat) of chemical compounds based on molecular descriptors. The dataset used for this task was provided through a Kaggle competition associated with the course, where our solution achieved a top-5 placement, ranking 3rd overall.

## 🔍 Approach
- Preprocessing with scaling and one-hot encoding
- Feature selection based on molecular descriptors
- Model tuning using Gradient Boosting Regressor and RandomizedSearchCV

## 🧰 Tools & Libraries
- Python, Pandas, NumPy, scikit-learn, Matplotlib
- Machine Learning Models: Linear Regression, Random Forest, Gradient Boosting

## 📁 Files
- train.csv, test.csv: Dataset files
- Data_exploration.ipynb: Exploratory analysis
- tester.py: Final model training and prediction script
- submission.csv: Final predictions for test data

## 🚀 Usage
Run  python tester.py in your favourite environment. The output/predictions will be generated in submission.csv
