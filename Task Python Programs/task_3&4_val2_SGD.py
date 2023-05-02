#%%
# ↑ Run as Cell

# Tasks 3 & 4 - Selecting a running a set of machine learning techniques to address the task 
# (Validation Method: K-Fold Cross | Model: Stochastic Gradient Descent)

# Imports:
import pandas as pd
from pandas import array
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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

# 1. Performing preprocessing scaling of the train and test data.
X_train_bike, X_test_bike, Y_train_bike, Y_test_bike = train_test_split(X_bike, Y_bike, test_size=0.2, random_state=3)

sc = StandardScaler()

X_train_bike = sc.fit_transform(X_train_bike)
X_test_bike = sc.transform(X_test_bike)


# 2. Validation Method: K-Fold Cross

#  Creating a KFold object with 5 folds.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initializing arrays to store the metrics for each fold.
mae_scores = []
mse_scores = []
r2_scores = []

# 3. Predicting using Stochastic Gradient Descent

# Defining the hyperparameters to be tuned.
param_grid0 = {
    'alpha': [1e-5, 1e-4, 1e-3],
    'max_iter': [1000, 2000, 3000],
    'tol': [1e-6, 1e-5, 1e-4, 1e-3],
    'eta0': [0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'optimal','adaptive']
}

# Creating a KFold object with 5 folds.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initializing arrays to store the metrics for each fold.
mae_scores = []
mse_scores = []
r2_scores = []

# Looping over the folds and fit a Stochastic Gradient Descent model to each training set, and evaluate on the corresponding test set.
for train_index, test_index in kf.split(X_train_bike):
    X_train, X_test = X_train_bike[train_index], X_train_bike[test_index]
    Y_train, Y_test = Y_train_bike.iloc[train_index], Y_train_bike.iloc[test_index]

    # Creating a grid search object with the Stochastic Gradient Descent, the hyperparameter grid, and other settings.
    stochastic_grad_val2 = GridSearchCV(
        SGDRegressor(), param_grid0, cv=3, scoring='neg_mean_absolute_error', verbose=0
    )

    # Fitting the grid search object to the training data.
    stochastic_grad_val2.fit(X_train, Y_train.values.ravel())

    # Predicting the target variable (bike rental count) using the fitted model and test set features.
    Y_pred = stochastic_grad_val2.predict(X_test)

    # Calculating the Mean Absolute Error between the predicted and actual bike rental counts in the test set.
    mae_scores.append(mean_absolute_error(Y_test, Y_pred))

    # Calculating the Mean Squared Error between the predicted and actual bike rental counts in the test set.
    mse_scores.append(mean_squared_error(Y_test, Y_pred))

    # Calculating the R-Squared score between the predicted and actual bike rental counts in the test set.
    r2_scores.append(r2_score(Y_test, Y_pred))

# Calculating the Mean Absolute Error from the 5-fold cross-validation.
print(f"Average MAE from 5-fold cross-validation: {np.mean(mae_scores)}")

# Calculating the Mean Squared Error from the 5-fold cross-validation.
print(f"Average MSE from 5-fold cross-validation: {np.mean(mse_scores)}")

# Calculating the R-Squared score from the 5-fold cross-validation.
print(f"Average R-squared from 5-fold cross-validation: {np.mean(r2_scores)}")
# %%
