# Football Prediction Model 4.0

A python machine learning model to predict the outcomes of EPL, La Liga, BundesLiga, Ligue One, and Serie A games.

<!-- Overview -->
<div id="overview"></div>

## Overview

A machine learning model designed to predict the outcome of football (soccer) matches. The model has been trained on data (140 data points) from the last 5 years, 2018-2023, predicting the expected goals for and against. Using a combination of those predictions and ELO ratings, frm current data, I then calculate offensive and defensive strengths for each team, with which I can predict the outcome of upcoming games, and determine whether my odds have an edge over Vegas.

This script also contains code for appending these predictions to a firebase real-time database.
   
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Built With -->
<div id="builtwith"></div>

## Built With
* [Python](https://python.org)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Dependencies -->
<div id="dependencies"></div>

 ## Dependencies
 - numpy
 - pandas
 - selenium
 - webdriver_manager
 - BeautifulSoup
 - seaborn
 - matplotlib.pyplot
 - sklearn
 - firebase_admin

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Acknowledgements -->
<div id="acknowledgements"></div>

 ## Acknowledgements
 
 Data sourced from FiveThirtyEight and FBref.

<p align="right">(<a href="#top">backop</a>)</p>
