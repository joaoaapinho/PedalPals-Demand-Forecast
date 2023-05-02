#%%
# ↑ Run as Cell

# Task 2 - Preparing the Dataset in the best possible manner.

# Imports:
import pandas as pd
import numpy as np

# Load the purchased Bike Sharing dataset:
bike_dataset = pd.read_csv("bike-sharing_hourly.csv", index_col=False)

#------------------------------Data Cleaning:----------------------------

# Check if there are null values.
bike_dataset.isna().astype(int).sum()
# Checked: There are no null values for any of the column attributes.


# Check if there are any Duplicated Entries.
bike_dataset.duplicated().sum()
# Checked: No entries are duplicated.


# Change 'dteday' datatype.
bike_dataset['dteday'] = bike_dataset['dteday'].astype('datetime64[ns]')

# Create a new 'day' of the month variable.
bike_dataset['day'] = bike_dataset.dteday.dt.day

# Set 'dteday' as index.
bike_dataset = bike_dataset.set_index('dteday')


# Creating a validation DataFrame to store all the 2012 Q4 values.
# Q4 Start and End Dates.
dateStart = '2012-10-01'
dateEnd = '2012-12-31'
bike_val_dataset = bike_dataset.loc[(bike_dataset.index >= dateStart) & (bike_dataset.index <= dateEnd)]

# Removing all entries from Q4 of 2012, in order to mimic the historical data that was purchased by the PedalPals.
bike_dataset = bike_dataset[(bike_dataset.index < dateStart) | (bike_dataset.index > dateEnd)]


# Removing outliers:
# Checking the entry values that are above 3rd + IQR * 1.5.
IQR = bike_dataset.cnt.quantile(0.75) - bike_dataset.cnt.quantile(0.25)
upper_fence = bike_dataset.cnt.quantile(0.75) + (IQR * 1.5)
outliers = bike_dataset[bike_dataset["cnt"] > upper_fence]

# Deleting the outliers from the dataset.
bike_dataset = bike_dataset.loc[bike_dataset.cnt <= upper_fence, :]

#-------------------------Feature Engineering:--------------------------

# Creating a new dataframe with a 'userType' additional column as casual.
dfCasual = bike_dataset.copy().reset_index()
dfCasual['userType'] = 'casual'

# Creating a new dataframe with a 'userType' additional column as registered.
dfReg = bike_dataset.copy().reset_index()
dfReg['userType'] = 'registered'

# Merging these two dataframes with an outer join.
dfUser = pd.merge(dfCasual, dfReg, how='outer')

# Assigning the value of the casual count to the 'cnt' column for all rows in the 'dfUser' dataframe where the 'userType' column equals "casual".
dfUser.loc[dfUser.userType == 'casual', 'cnt'] = dfUser['casual']
# Assigning the value of the registered count to the 'cnt' column for all rows in the 'dfUser' dataframe where the 'userType' column equals "registered".
dfUser.loc[dfUser.userType == 'registered', 'cnt'] = dfUser['registered']

# Removing unecessary columns and displaying the first rows of the resulting dataframe.
dfUser = dfUser.drop(['casual','registered'], axis=1)
dfUser.sort_values(by='instant').head(6)


# 1. Creating a new feature 'dayInstant' with 4 different time periods.
def dayInstant(hr):
    if 6 <= hr and hr <= 9: return 'morning'
    elif 10 <= hr and hr <= 17: return 'midday'
    elif 18 <= hr and hr <= 21: return 'afternoon'
    else: return 'night'

bike_dataset['dayInstant'] = bike_dataset['hr'].apply(lambda x: dayInstant(x))
dfUser['dayInstant'] = dfUser['hr'].apply(lambda x: dayInstant(x))


# 2. Adding the percentage of registered bikes over total count feature.
bike_dataset['pct_registered'] = bike_dataset['registered']/bike_dataset['cnt']

# Aggregating the percentage of registered bikes over total count feature on a hourly basis.
pct_registered_hourly = bike_dataset.groupby('hr')['pct_registered'].mean().to_dict()
bike_dataset['pct_registered_hourly'] = bike_dataset['hr'].map(pct_registered_hourly)

# Aggregating the percentage of registered bikes over total count feature on a monthly basis.
pct_registered_monthly = bike_dataset.groupby('mnth')['pct_registered'].mean().to_dict()
bike_dataset['pct_registered_monthly'] = bike_dataset['mnth'].map(pct_registered_monthly)


# 3. Adding a night and day feature.
bike_dataset["day_night"] = bike_dataset['hr'].apply(lambda x: 1 if x >= 7 and x <= 23 else 0)
bike_dataset.groupby('day_night')['cnt'].mean()


# 4. Adding 'temp*windspeed' and 'hum^2' features to the DataFrame.
bike_dataset['temp*windspeed'] = bike_dataset['temp']*bike_dataset['windspeed']
bike_dataset['hum^2'] = np.square(bike_dataset['hum'])

# 5. Dropping 'atemp' and 'hum' features given their high correlation with other variables and risk of multicollinearity. 
bike_dataset = bike_dataset.drop(['atemp', 'hum'], axis=1)
#----------------------------------------------------------------------
# Updated bike_dataset.
bike_dataset
# %%