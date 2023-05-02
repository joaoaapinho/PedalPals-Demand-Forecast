#%%
# ↑ Run as Cell

# Task 1 - Selecting the Bike Sharing Dataset

# Problem:
# "Can PedalPals reliably forecast Q4 2012 shared bike demand in the US to decide whether to enter the market during that period?"

# Main Constraint: PedalPals will only decide to expand on Q4 if the total market number of rides for that quarter is projected to exceed 300,000.

# Imports:
import pandas as pd
import numpy as np

# Load the purchased Bike Sharing dataset:

bike_dataset = pd.read_csv("bike-sharing_hourly.csv", index_col=False)

print(bike_dataset)
# %%
