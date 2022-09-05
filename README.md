## Regression model test task
Task: You have a dataset (internship_train.csv) that contains 53 anonymized features and a target column. Your task is to build model that predicts a target based on the proposed features. Please provide predictions for internship_hidden_test.csv file. Target metric is RMSE. The main goal is to provide github repository that contains

### Main files
Quantum3_Analysis.ipynb -- IPython file with exploratory analysis and finding best model  
main.py -- Python file with model training and prediction  
internship_test.csv -- CSV file with prediction results  

### Model
While trying to solve the problem I tried various regressor models like simple linear regression, k-nearest neighbours regressor and random forest but after some analysis I came up with Polynomial Regression model with only 2 of 53 features and it performed incredibly well with only 8e-13 RMSE score
