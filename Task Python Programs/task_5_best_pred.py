#%%
# ↑ Run as Cell

# Tasks 5 - Most suitable solution (best performing model)
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

# ---------------------Generate Evaluation Dataset-----------------------

# Load the purchased Bike Sharing dataset:
bike_val_dataset = pd.read_csv("bike_val_dataset.csv",index_col='dteday')

# Load the X_bike dataset:
X_bike = pd.read_csv("X_bike_dataset.csv", index_col='dteday')

# Load the Y_bike dataset:
Y_bike = pd.read_csv("Y_bike_dataset.csv", index_col='dteday')

# 1. Performing the same transformations as bike_dataset

# Transformation 1: Creating a new variable 'dayInstant' with 4 different time periods.

def dayInstant(hr):
    if 6 <= hr and hr <= 9: return 'morning'
    elif 10 <= hr and hr <= 17: return 'midday'
    elif 18 <= hr and hr <= 21: return 'afternoon'
    else: return 'night'

bike_val_dataset['dayInstant'] = bike_val_dataset['hr'].apply(lambda x: dayInstant(x))


# Transformation 2: Adding the percentage of registered bikes over total count feature.

bike_val_dataset['pct_registered'] = bike_val_dataset['registered']/bike_val_dataset['cnt']

# Aggregating the percentage of registered bikes over total count feature on a hourly basis.
pct_registered_hourly = bike_val_dataset.groupby('hr')['pct_registered'].mean().to_dict()
bike_val_dataset['pct_registered_hourly'] = bike_val_dataset['hr'].map(pct_registered_hourly)

# Aggregating the percentage of registered bikes over total count feature on a monthly basis.
pct_registered_monthly = bike_val_dataset.groupby('mnth')['pct_registered'].mean().to_dict()
bike_val_dataset['pct_registered_monthly'] = bike_val_dataset['mnth'].map(pct_registered_monthly)

# Transformation 3: Adding the night and day feature.

bike_val_dataset["day_night"] = bike_val_dataset['hr'].apply(lambda x: 1 if x >= 7 and x <= 23 else 0)

# Transformation 4: Adding the 'temp*windspeed' and 'hum^2' features to the DataFrame and dropping 'atemp' and 'hum' features.

bike_val_dataset['temp*windspeed'] = bike_val_dataset['temp']*bike_val_dataset['windspeed']
bike_val_dataset['hum^2'] = np.square(bike_val_dataset['hum'])

bike_val_dataset = bike_val_dataset.drop(['atemp', 'hum'], axis=1)


#------------------------------Data Preparation----------------------------

# - Defining the independent (pred_bike_data) features.

pred_bike_data = bike_val_dataset.loc[:,[
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

# Converting categorical features into dummy values.
pred_bike_data = pd.get_dummies(pred_bike_data, columns=['season','mnth','weekday','weathersit','dayInstant'])


# Getting the list of columns that exist in X_bike but not in X_pred_bike_val.
missing_columns = set(X_bike.columns) - set(pred_bike_data.columns)


# Adding dummy columns to pred_bike_data with value 0 for missing columns.
for col in missing_columns:
    pred_bike_data[col] = 0


# Sorting the columns to be in the same order as X_bike.

# Getting the list of column names from X_bike.
bike_columns = X_bike.columns

# Reordering the columns in X_pred_bike_val to match the order in X_bike.
pred_bike_data = pred_bike_data[bike_columns]

#-------------------------------Data Split-------------------------------

# The original dataset will now serve as the training set.
X_train_bike = X_bike
Y_train_bike = Y_bike #Target

# The validation set.
X_pred_bike_final = pred_bike_data


#-------------------------------Prediction-------------------------------

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
xgboost_val1.fit(X_train_bike, Y_train_bike.values.ravel())

# Predicting the target variable (bike rental count) using the fitted model and test set features.
Y_pred_bike_final = xgboost_val1.predict(X_pred_bike_final)

# Adding the target value prediction to the prediction DataFrame.
pred_bike_data['cnt'] = Y_pred_bike_final

# Concatenating X_bike and Y_bike along columns.
train_bike_dataset = pd.concat([X_bike, Y_bike], axis=1)

# Concatenating train_bike_data and pred_bike_dataset along rows.
full_bike_dataset = pd.concat([train_bike_dataset, pred_bike_data], axis=0)

# Outputting the full dataset (training + prediction).
full_bike_dataset

#--------------------------Q4 Demand Computation--------------------------

# Filtering the dataset for Q4 of 2012.
q4_2012_data = full_bike_dataset['2012-10-01':'2012-12-31']

# Calculating the expected total number of rides in Q4 of 2012.
total_rides_q4_2012 = q4_2012_data['cnt'].sum().round()

print(f"Total number of rides in Q4 of 2012: {int(total_rides_q4_2012)} rides.")
# %%
