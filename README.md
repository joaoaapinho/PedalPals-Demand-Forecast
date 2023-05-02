<p align="center">
  <img src="https://user-images.githubusercontent.com/114337279/235669152-78e42812-ce23-4d29-9c97-431d8873e817.png" alt="Small logo" width="30%">
</p>
<h3 align="center">PedalPals: Bike Sharing Demand Forecasting</h3>

<p align="center"><b>Done by:</b> João André Pinho</p>

<h2> 👁‍🗨 Overview </h2>

- **🏢 Business Case**

**PedalPals**, a fictious Dutch bike-sharing company, is considering expanding into the US market. To make an informed decision, PedalPals' Machine Learning and AI Engineering Team proposed estimating the demand for bike rentals in Q4 of 2012 using an acquired dataset on the US bike-sharing scene from a reputable market research firm.

- **❔ Problem Definition**

**"Can PedalPals reliably forecast Q4 2012 shared bike demand in the US to decide whether to enter the market during that period?"**

The project goal is to forecast whether the **total market demand for bike rentals in Q4 of 2012 in the US is projected to meet or exceed PedalPals' requirement of 300,000 rides**, which is a prerequisite for the company's planned entry into the market during that period. Upon expansion, PedalPals aims to capture 5% market share immediately.

<h2> 💻 Technology Stack </h2>

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Plotly.

<h2> 🧮 Dataset </h2>

The dataset that was used contains a comprehensive collection of bike sharing data from 2011 and an estimation for the first three quarters of 2012 that captures a variety of characteristics that impact bike utilization.

``` ***Dataset Attribute Information:***

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
```
