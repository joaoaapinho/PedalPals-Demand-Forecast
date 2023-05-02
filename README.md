<p align="center">
  <img src="https://user-images.githubusercontent.com/114337279/235669152-78e42812-ce23-4d29-9c97-431d8873e817.png" alt="Small logo" width="30%">
</p>
<h3 align="center">PedalPals: Bike Sharing Demand Forecasting</h3>

<p align="center"><b>Done by:</b> João André Pinho</p>

<h2> 👁‍🗨 Overview </h2>

<h3>🏢 Business Case</h3>

**PedalPals**, a fictious Dutch bike-sharing company, is considering expanding into the US market. To make an informed decision, PedalPals' Machine Learning and AI Engineering Team proposed estimating the demand for bike rentals in Q4 of 2012 using an acquired dataset on the US bike-sharing scene from a reputable market research firm.

<h3>❔ Problem Definition</h3>

**"Can PedalPals reliably forecast Q4 2012 shared bike demand in the US to decide whether to enter the market during that period?"**

The project goal is to forecast whether the **total market demand for bike rentals in Q4 of 2012 in the US is projected to meet or exceed PedalPals' requirement of 300,000 rides**, which is a prerequisite for the company's planned entry into the market during that period. Upon expansion, PedalPals aims to capture 5% market share immediately.

<h2> 💻 Technology Stack </h2>

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Plotly.

<h2> 🧮 Dataset </h2>

The dataset that was used contains a comprehensive collection of bike sharing data from 2011 and an estimation for the first three quarters of 2012 that captures a variety of characteristics that impact bike utilization.

**Dataset Features:**

- `instant`: Record index
- `dteday`: Record date
- `season`: Season (1-Spring, 2-Summer, 3-Fall, 4-Winter)
- `yr`: Year (0-2011, 1-2012)
- `mnth`: Month (1-12)
- `hr`: Hour (0-23)
- `holiday`: Whether a day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
- `weekday`: Day of the week
- `workingday`: If a day is neither a weekend nor holiday - 1, otherwise - 0
- `weathersit`: 
  - 1 - Clear, Few clouds, Partly cloudy, Partly cloudy
  - 2 - Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  - 3 - Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
  - 4 - Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp`: Normalized temperature in Celsius. The values are up to 41 (max)
- `atemp`: Normalized feeling temperature in Celsius. The values are up to 50 (max)
- `hum`: Normalized humidity. The values are up to 100 (max)
- `windspeed`: Normalized wind speed. The values are up to 67 (max)
- `casual`: Count of casual users
- `registered`: Count of registered users
- `cnt`: Count of total rental bikes (including both casual and registered)

<h2> 🤖 Main Models and Techniques </h2>

- **Linear Regression**
- **Stochastic Gradient Descent Regression**
- **Decision Tree Regression**
- **Random Forest Regression**
- **XGBoost Regression**
- **Multi-layer Perceptron Regression**
- **Leave-One-Out Cross-Validation**
- **K-Fold Cross-Validation**
- **Feature Scaling using StandardScaler**
- **Principal Component Analysis (PCA)**
- **Polynomial Feature Transformation**
- **One-Hot Encoding using OrdinalEncoder**
- **Statistical Analysis using f_classif**
- **Model Evaluation using Mean Absolute Error, Mean Squared Error, R-squared Score**

**Note:** For a detailed description of the methodology and the reasoning behind the decisions made, please refer to the accompanying Jupyter Notebook.

<h2> 🎯 Conclusions and Limitations </h2>

<h3> Conclusions </h3>

The application of machine learning models has provided valuable insights for addressing the problem of forecasting bike rental demand.


<h4>📊 Models' Performance Results:</h4>

Through **Train & Test Split Validation:**


**MAE:**

| Linear Regression | Stochastic Gradient Descent |Decision Tree Regressor | Random Forest Regressor | XGBoost | MLP Regression |
| --- | --- | --- | --- | --- | --- |
| 74.12 | 73.98 | 56.13 | 38.89 | 34.18 | 40.55 |

**MSE:**

| Linear Regression | Stochastic Gradient Descent |Decision Tree Regressor | Random Forest Regressor | XGBoost | MLP Regression |
| --- | --- | --- | --- | --- | --- |
| 9827.83 | 9810.06 | 7338.52 | 3338.64 | 2699.85 | 3557.95 |

**R-Squared:**

| Linear Regression | Stochastic Gradient Descent |Decision Tree Regressor | Random Forest Regressor | XGBoost | MLP Regression |
| --- | --- | --- | --- | --- | --- |
| 0.57 | 0.57 | 0.68 | 0.86 |  0.88 | 0.85 |


Through **K-Fold Cross Validation**:


**MAE:**

| Linear Regression | Stochastic Gradient Descent |Decision Tree Regressor | Random Forest Regressor | XGBoost | MLP Regression |
| --- | --- | --- | --- | --- | --- |
| 74.54 | 77.15 | 52.70 | 40.57 | 34.85 | 41.84 |

**MSE:**

| Linear Regression | Stochastic Gradient Descent |Decision Tree Regressor | Random Forest Regressor | XGBoost | MLP Regression |
| --- | --- | --- | --- | --- | --- |
| 9843.35 | 10448.84 | 6337.92 | 3542.45 | 2771.74 | 3636.73 |

**R-Squared:**

| Linear Regression | Stochastic Gradient Descent |Decision Tree Regressor | Random Forest Regressor | XGBoost | MLP Regression |
| --- | --- | --- | --- | --- | --- |
| 0.57 | 0.55 | 0.72 | 0.85 | 0.88 | 0.84 |


<h4>🏆 Best Model:</h4>
The best model in this case was the **XGBoost regressor**. Theoretically speaking, XGBoost is known for its ability to handle a wide range of data types and relationships within the dataset. It is a method of **ensemble learning that integrates numerous weak models (decision trees) to produce a more accurate and robust model**. By iteratively adding new trees to the ensemble and minimizing the loss function by optimizing its parameters through gradient boosting, this method allows the model to **capture complex, non-linear patterns in the data and enhance its prediction performance.**


<h4>🥈 Other Models:</h4>
In comparison, other models such as **Linear Regression and MLP Regression**, are based on **simpler assumptions regarding variable relationships**. Linear Regression is based on the assumption of a linear relationship between the features and the target variable, which may not necessarily be true in real-world datasets. MLP Regression, while more flexible than Linear Regression, is dependent on neural network design and may require substantial tuning to obtain good results. Tree-based models, on the other hand, such as **Decision Tree Regressor and Random Forest Regressor**, offer the benefit of managing non-linear connections and feature interactions organically. Nevertheless, Decision Trees are prone to overfitting, whereas **Random Forest**, while more resilient, **still fell short of XGBoost in terms of prediction performance**.


<h4> 🛫 Business Results:</h4>
While being capable of predicting the **total demand for Q4 of 2012: 317328 rides**, PedalPals managers gained valuable information that contributed towards their decision to internationalize.

By estimating that they would capture 5% of these rides immediatly upon expansion, other important analysis such as revenues planning, required equipment investment, among others will also possible.

Notwithstanding the predicted **negative trend shown in US shared bike demand in Q4 of 2012**, this is a reasonable evolution (that also happened in the previous year) since with cooler temperatures and probably rain, people tend to ride less bikes. In any case, the fact that the predicted ride threshold of 300,000 rides was met gives the firm confidence that its internationalization in Q4 will be data-driven and well-supported.



<h3> Limitations </h3>
While this project gave significant insights on anticipating bike rental demand to make an important internationalization decision, there are certain limitations that should be acknowledged:


- **🔧 Feature Engineering:**
Although some new transformation variables were added to the dataset, including some polynomial ones, the current set of features used for modeling may not have captured all the relevant information affecting bike rental demand. For future improvements, additional features like competing transportation options or other combinations of already existing ones, could be considered to improve the model's predictive power.


- **🧰 Hyperparameter Tuning:**
The models' performance might be enhanced by conducting a more comprehensive search for optimum hyperparameters. Although Grid Search CV was used, other ones such as Random Search or Tune Grid Search might also be employed to determine better combinations of hyperparameters for each model, perhaps leading to improved performance.


- **⌚ Temporal Aspects:** 
Time series data include underlying temporal patterns that may have gone unnoticed in the current modeling technique. To explicitly handle the time-dependent character of the data, neural networks such as ARIMA, or LSTM (Long Short-Term Memory) may be explored in the future.


- **🔮 Prediction Process:**
Considering that certain features in the prediction data were real and that some of these were correlated with the target column `cnt` (e.g., pct_registered), the final prediction results may be somewhat biased. While the developed reasoning and prediction process that were used are valid for the proposed exercise, using only variables in the prediction dataset that would have been estimated by other models could have led to more impartial results.


- **🤔 Model Interpretability:**
Complex models, such as the XGBoost, may be more difficult to understand than simpler models, such as a Linear Regression. It is vital to set a balance between prediction accuracy and interpretability, especially when the goal is to provide actionable information to decision-makers.

All in all, future modelling efforts from PedalPals might possibly generate even better results in terms of both forecast performance and general understanding of the variables influencing bike rental demand in the US, by addressing some of these limitations and/or adopting alternate methodologies.
