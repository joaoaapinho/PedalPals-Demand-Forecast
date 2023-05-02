#%%
# ↑ Run as Cell

# Tasks 3 & 4 - Selecting a running a set of machine learning techniques to address the task
# (Validation Method: Train and Test Split | Model: XGBoost Regressor)

# Imports:
import pandas as pd
from pandas import array
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

# 3.1. Using PCA (with XGBoost) with 1 principal component. --------------------------------------------------

# Creating a PCA instance that will reduce the dimensionality of the dataset to 1 component.
pca = PCA(n_components=1)

# Applying PCA to the training data to reduce its dimensionality.
X_train_bike_pca1 = pca.fit_transform(X_train_bike)

# Applying the same PCA transformation to the test data.
X_test_bike_pca1 = pca.transform(X_test_bike)

# Getting the variance explained by each principal component.
explained_variance = pca.explained_variance_ratio_

# Outputing the variance explained by each principal component.
explained_variance

# Defining a dictionary with the hyperparameter values to be tested.
param_grid3 = {
    'n_estimators': [100,120],
    'max_depth':[8,9,10,15],
    'learning_rate':[0.1,0.15,0.20],
    }

# Creating a grid search object with the XGBRegressor, the hyperparameter grid, and other settings.
xgboost_val1 = GridSearchCV(
    XGBRegressor(), param_grid3, cv=3, scoring='neg_mean_absolute_error', verbose=0
)

# Fitting the grid search object to the training data.
xgboost_val1.fit(X_train_bike_pca1, Y_train_bike.values.ravel())

# Predicting the target variable (bike rental count) using the fitted model and test set features.
Y_pred_bike1 = xgboost_val1.predict(X_test_bike_pca1)

print("PCA Results with 1 Principal Component:\n")
# Calculating the Mean Absolute Error between the predicted and actual bike rental counts in the test set.
print(f"MAE from TestSet: {mean_absolute_error(Y_test_bike, Y_pred_bike1)}")

# Calculating the Mean Squared Error between the predicted and actual bike rental counts in the test set.
print(f"MSE from TestSet: {mean_squared_error(Y_test_bike, Y_pred_bike1)}")

# Calculating the R-Squared score between the predicted and actual bike rental counts in the test set.
print(f"R-squared from TestSet: {r2_score(Y_test_bike, Y_pred_bike1)}")


# 3.2. Using PCA (with XGBoost) with 1 principal components. --------------------------------------------------

# Creating a PCA instance that will reduce the dimensionality of the dataset to 1 component.
pca = PCA(n_components=2)

# Applying PCA to the training data to reduce its dimensionality.
X_train_bike_pca1 = pca.fit_transform(X_train_bike)

# Applying the same PCA transformation to the test data.
X_test_bike_pca1 = pca.transform(X_test_bike)

# Getting the variance explained by each principal component.
explained_variance = pca.explained_variance_ratio_

# Outputing the variance explained by each principal component.
explained_variance

# Defining a dictionary with the hyperparameter values to be tested.
param_grid3 = {
    'n_estimators': [100,120],
    'max_depth':[8,9,10,15],
    'learning_rate':[0.1,0.15,0.20],
    }

# Creating a grid search object with the XGBRegressor, the hyperparameter grid, and other settings.
xgboost_val1 = GridSearchCV(
    XGBRegressor(), param_grid3, cv=3, scoring='neg_mean_absolute_error', verbose=0
)

# Fitting the grid search object to the training data.
xgboost_val1.fit(X_train_bike_pca1, Y_train_bike.values.ravel())

# Predicting the target variable (bike rental count) using the fitted model and test set features.
Y_pred_bike1 = xgboost_val1.predict(X_test_bike_pca1)

print("\nPCA Results with 2 Principal Components:\n")
# Calculating the Mean Absolute Error between the predicted and actual bike rental counts in the test set.
print(f"MAE from TestSet: {mean_absolute_error(Y_test_bike, Y_pred_bike1)}")

# Calculating the Mean Squared Error between the predicted and actual bike rental counts in the test set.
print(f"MSE from TestSet: {mean_squared_error(Y_test_bike, Y_pred_bike1)}")

# Calculating the R-Squared score between the predicted and actual bike rental counts in the test set.
print(f"R-squared from TestSet: {r2_score(Y_test_bike, Y_pred_bike1)}")


# 3.3. Using PCA (with XGBoost) with 6 principal components. --------------------------------------------------

# Creating a PCA instance that will reduce the dimensionality of the dataset to 1 component.
pca = PCA(n_components=6)

# Applying PCA to the training data to reduce its dimensionality.
X_train_bike_pca1 = pca.fit_transform(X_train_bike)

# Applying the same PCA transformation to the test data.
X_test_bike_pca1 = pca.transform(X_test_bike)

# Getting the variance explained by each principal component.
explained_variance = pca.explained_variance_ratio_

# Outputing the variance explained by each principal component.
explained_variance

# Defining a dictionary with the hyperparameter values to be tested.
param_grid3 = {
    'n_estimators': [100,120],
    'max_depth':[8,9,10,15],
    'learning_rate':[0.1,0.15,0.20],
    }

# Creating a grid search object with the XGBRegressor, the hyperparameter grid, and other settings.
xgboost_val1 = GridSearchCV(
    XGBRegressor(), param_grid3, cv=3, scoring='neg_mean_absolute_error', verbose=0
)

# Fitting the grid search object to the training data.
xgboost_val1.fit(X_train_bike_pca1, Y_train_bike.values.ravel())

# Predicting the target variable (bike rental count) using the fitted model and test set features.
Y_pred_bike1 = xgboost_val1.predict(X_test_bike_pca1)

print("\nPCA Results with 6 Principal Components:\n")
# Calculating the Mean Absolute Error between the predicted and actual bike rental counts in the test set.
print(f"MAE from TestSet: {mean_absolute_error(Y_test_bike, Y_pred_bike1)}")

# Calculating the Mean Squared Error between the predicted and actual bike rental counts in the test set.
print(f"MSE from TestSet: {mean_squared_error(Y_test_bike, Y_pred_bike1)}")

# Calculating the R-Squared score between the predicted and actual bike rental counts in the test set.
print(f"R-squared from TestSet: {r2_score(Y_test_bike, Y_pred_bike1)}")


# 3.4. Using PCA (with XGBoost) with n principal components in order to explain 90% of the variance. --------------------------------------------------

# Creating a PCA instance that will reduce the dimensionality of the dataset to the necessary number of components (n) in order to explain 90% of the variance of the original dataset.
pca = PCA(n_components=0.9)

# Applying PCA to the training data to reduce its dimensionality.
X_train_bike_pca1 = pca.fit_transform(X_train_bike)

# Applying the same PCA transformation to the test data.
X_test_bike_pca1 = pca.transform(X_test_bike)

# Getting the variance explained by each principal component.
explained_variance = pca.explained_variance_ratio_

# Outputing the variance explained by each principal component.
explained_variance

# Defining a dictionary with the hyperparameter values to be tested.
param_grid3 = {
    'n_estimators': [100,120],
    'max_depth':[8,9,10,15],
    'learning_rate':[0.1,0.15,0.20],
    }

# Creating a grid search object with the XGBRegressor, the hyperparameter grid, and other settings.
xgboost_val1 = GridSearchCV(
    XGBRegressor(), param_grid3, cv=3, scoring='neg_mean_absolute_error', verbose=0
)

# Fitting the grid search object to the training data.
xgboost_val1.fit(X_train_bike_pca1, Y_train_bike.values.ravel())

# Predicting the target variable (bike rental count) using the fitted model and test set features.
Y_pred_bike1 = xgboost_val1.predict(X_test_bike_pca1)

print("\nPCA Results with n Principal Components to explain 90% of the Variance:\n")
# Calculating the Mean Absolute Error between the predicted and actual bike rental counts in the test set.
print(f"MAE from TestSet: {mean_absolute_error(Y_test_bike, Y_pred_bike1)}")

# Calculating the Mean Squared Error between the predicted and actual bike rental counts in the test set.
print(f"MSE from TestSet: {mean_squared_error(Y_test_bike, Y_pred_bike1)}")

# Calculating the R-Squared score between the predicted and actual bike rental counts in the test set.
print(f"R-squared from TestSet: {r2_score(Y_test_bike, Y_pred_bike1)}")

# %%
