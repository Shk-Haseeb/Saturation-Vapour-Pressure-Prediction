## Predicting Saturation Vapour Pressure (pSat)

### Project Overview
This project was developed as part of the Introduction to Machine Learning course at the University of Helsinki. The aim was to build a predictive model for estimating the saturation vapour pressure (pSat) of chemical compounds based on molecular descriptors. The dataset used for this task was provided through a Kaggle competition associated with the course, where our solution achieved a top-5 placement, ranking 3rd overall.

### Approach
- Preprocessed data with scaling & one-hot encoding  
- Selected features based on molecular structure  
- Tuned a **Gradient Boosting Regressor** using **RandomizedSearchCV**

### Tech Stack  
**Languages/Tools**: Python, pandas, NumPy, scikit-learn, Matplotlib  
**ML Models**: Linear Regression, Random Forest, Gradient Boosting

### Key Files  
- `Data_exploration.ipynb` – EDA & insights  
- `tester.py` – Final model & predictions  
- `submission.csv` – Output predictions  
- `train.csv`, `test.csv` – Provided datasets

## Usage
Run  python tester.py in your favourite environment. The output/predictions will be generated in submission.csv
