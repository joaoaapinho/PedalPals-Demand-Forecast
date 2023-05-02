
# Task 6 - Informed reflection on how the application of machine learning solutions provideinsights useful for addressing the underlying problem.

# /---------------\
# • Conclusions:
# \---------------/

# The application of machine learning models has provided valuable insights for addressing the problem of forecasting bike rental demand.

#-------------------------
#- **Best Model:** 
#-------------------------

# The best model in this case was the XGBoost Regressor. Theoretically speaking, XGBoost is known for its ability to handle a wide range of data 
# types and relationships within the dataset. It is a method of ensemble learning that integrates numerous weak models (decision trees) to produce 
# a more accurate and robust model. By iteratively adding new trees to the ensemble and minimizing the loss function by optimizing its parameters 
# through gradient boosting, this method allows the model to capture complex, non-linear patterns in the data and enhance its prediction performance.

#-------------------------
#- **Other Models:**
#-------------------------

# In comparison, other models such as Linear Regression and MLP Regression, are based on simpler assumptions regarding variable relationships. 
# Linear Regression is based on the assumption of a linear relationship between the features and the target variable, which may not necessarily be true 
# in real-world datasets. MLP Regression, while more flexible than Linear Regression, is dependent on neural network design and may require substantial 
# tuning to obtain good results.

# Tree-based models, on the other hand, such as Decision Tree Regressor and Random Forest Regressor, offer the benefit of managing non-linear connections 
# and feature interactions organically. Nevertheless, Decision Trees are prone to overfitting, whereas Random Forest, while more resilient, still fell 
# short of XGBoost in terms of prediction performance.

#-------------------------
#- **Business Results:**
#-------------------------

# While being capable of predicting the total demand for Q4 of 2012: 317328 rides, PedalPals managers gained valuable information that contributed towards 
# their decision to internationalize.

# By estimating that they would capture 5% of these rides immediatly upon expansion, other important analysis such as revenues planning, required equipment 
# investment, among others will also possible.

# Notwithstanding the predicted negative trend shown in US shared bike demand in Q4 of 2012, this is a reasonable evolution (that also happened in the previous 
# year) since with cooler temperatures and probably rain, people tend to ride less bikes. In any case, the fact that the predicted ride threshold of 300,000 rides
# was met gives the firm confidence that its internationalization in Q4 will be data-driven and well-supported.


# /---------------\
# • Limitations:
# \---------------/

# While this business case gave significant insights on anticipating bike rental demand to make an important internationalization decision, there are certain 
# limitations that should be acknowledged:

#-----------------------------
#- **Feature Engineering:** 
#-----------------------------

# Although some new transformation variables were added to the dataset, including some polynomial ones, the current set of features used for modeling may not have
# captured all the relevant information affecting bike rental demand. For future improvements, additional features like competing transportation options or other 
# combinations of already existing ones, could be considered to improve the model's predictive power.

#-----------------------------
##- **Hyperparameter Tuning:** 
#-----------------------------

# The models' performance might be enhanced by conducting a more comprehensive search for optimum hyperparameters. Although Grid Search CV was used, other ones such 
# as Random Search or Tune Grid Search might also be employed to determine better combinations of hyperparameters for each model, perhaps leading to improved performance.

#-----------------------------
#- **Temporal Aspects:** 
#-----------------------------

# Time series data include underlying temporal patterns that may have gone unnoticed in the current modeling technique. To explicitly handle the time-dependent character 
# of the data, neural networks such as ARIMA, or LSTM (Long Short-Term Memory) may be explored in the future.

#-----------------------------
#- **Prediction Method:**
#-----------------------------

# Considering that certain features in the prediction data were real and that some of these were correlated with the target column `cnt` (e.g., pct_registered), the final 
# prediction results may be somewhat biased. While the developed reasoning and prediction process that were used are valid for the proposed exercise, using only variables 
# in the prediction dataset that would have been estimated by other models could have led to more impartial results.

#-----------------------------
#- **Model Interpretability:**
#-----------------------------

# Complex models, such as the XGBoost, may be more difficult to understand than simpler models, such as a Linear Regression. It is vital to set a balance between 
# prediction accuracy and interpretability, especially when the goal is to provide actionable information to decision-makers.


# All in all, future modelling efforts from PedalPals might possibly generate even better results in terms of both forecast performance and general understanding of 
# the variables influencing bike rental demand in the US, by addressing some of these limitations and/or adopting alternate methodologies.