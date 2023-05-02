#%%
# ↑ Run as Cell

# Tasks 3 & 4 - Selecting a running a set of machine learning techniques to address the task
# (Validation Method: Train and Test Split | Model: Stochastic Gradient Descent)

# Imports:
import pandas as pd
from pandas import array
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Loading the cleaned bike_dataset from task 2.
bike_dataset = pd.read_csv('bike_dataset_cleaned.csv',index_col='dteday')

#------------------------------Data Preparation----------------------------

# - Defining the independent (X_bike) features.

X_bike = bike_dataset.loc[:,[
    'season',
    'mnth',
    'hr',
    'holiday',
    'weekday',
    'workingday',
    'weathersit',
    'temp',
    'hum^2',
    'windspeed',
    'temp*windspeed',
    'dayInstant',
    'day_night',
    'pct_registered_hourly',
    'pct_registered_monthly'
]]

# - Defining the target column (Y_bike).
Y_bike = bike_dataset.loc[:,['cnt']]


# Converting categorical features into dummy values.
X_bike = pd.get_dummies(X_bike, columns=['season','mnth','weekday','weathersit','dayInstant'])

#------------------------------Prediction & Tuning----------------------------

# 1. Validation Method: Train & Test Split:

# Performing a train and test split.
X_train_bike, X_test_bike, Y_train_bike, Y_test_bike = train_test_split(X_bike, Y_bike, test_size=0.2, random_state=3)


# 2. Performing preprocessing scaling of the train and test data.
sc = StandardScaler()

X_train_bike = sc.fit_transform(X_train_bike)
X_test_bike = sc.transform(X_test_bike)

# 3. Predicting using Stochastic Gradient Descent.

# Defining the hyperparameters to be tuned.
param_grid0 = {
    'alpha': [1e-5, 1e-4, 1e-3],
    'max_iter': [1000, 2000, 3000],
    'tol': [1e-6, 1e-5, 1e-4, 1e-3],
    'eta0': [0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'optimal','adaptive']
}

# Creating a grid search object with the Stochastic Gradient Descent, the hyperparameter grid, and other settings.
stochastic_grad_val1 = GridSearchCV(
    SGDRegressor(), param_grid0, cv=3, scoring='neg_mean_absolute_error', verbose=0
)

# Fitting the grid search object to the training data.
stochastic_grad_val1.fit(X_train_bike, Y_train_bike.values.ravel())

# Predicting the target variable (bike rental count) using the fitted model and test set features.
Y_pred_bike = stochastic_grad_val1.predict(X_test_bike)

# Calculating the Mean Absolute Error between the predicted and actual bike rental counts in the test set.
print(f"MAE from TestSet: {mean_absolute_error(Y_test_bike, Y_pred_bike)}")

# Calculating the Mean Squared Error between the predicted and actual bike rental counts in the test set.
print(f"MSE from TestSet: {mean_squared_error(Y_test_bike, Y_pred_bike)}")

# Calculating the R-Squared score between the predicted and actual bike rental counts in the test set.
print(f"R-squared from TestSet: {r2_score(Y_test_bike, Y_pred_bike)}")
# %%
