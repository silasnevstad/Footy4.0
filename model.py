# FOOTY MODEL 4.0
import pandas as pd

# data scraping
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import io

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# =============================================== [--- Downloading data ---] ===============================================
# since fbref provides the data not in per 90 format, we need to use selenium to download the data instead of just using requests and soup

opts = Options()
# opts.add_argument("--headless")
# opts.add_experimental_option("excludeSwitches", ["enable-automation"])
# opts.add_experimental_option('useAutomationExtension', False)

# def fastDownload(url):
#     html_data = requests.get(url).content
    
#     soup = BeautifulSoup(html_data, 'html.parser')
    
#     ids = ['stats_squads_standard_for_per_match_toggle',
#            'stats_squads_keeper_for_per_match_toggle',
#            'stats_squads_keeper_adv_for_per_match_toggle',
#            'stats_squads_shooting_for_per_match_toggle',
#            'stats_squads_passing_for_per_match_toggle',
#            'stats_squads_gca_for_per_match_toggle',
#            'stats_squads_defense_for_per_match_toggle',
#            'stats_squads_possession_for_per_match_toggle',
#            'stats_squads_playing_time_for_per_match_toggle']
    
#     for id in ids:
#         table = soup.find('table', {'id': id})
#         df = pd.read_html(str(table))[0]
#         # how can i check how many header levels there are? answer: df.columns.nlevels
#         if df.columns.nlevels == 2:
#             df.columns = df.columns.droplevel()
        
#         # some of the columns might not be in 90 min format, so we need to check that, and if not, convert it
#         matches_played = df['MP']
#         for column in df.columns:
#             if column != 'Squad' and column != 'MP':
#                 # if the column does not contain 
#                 df[column] = df[column].apply(lambda x: x / matches_played)   

def downloadData(url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = opts) # downloads and sets newest chromedriver
    driver.get(url) # driver launches the given url (fbref EPL)
    driver.maximize_window()
    
    closeAdButton = driver.find_element(By.CSS_SELECTOR, '[class*="fs-close-button fs-close-button-sticky"]')
    
    toggles = []
    
    def addToggles(id):
        try:
            toggle = driver.find_element(By.ID, id)
            toggles.append(toggle)
        except:
            pass
    
    addToggles("stats_squads_standard_for_per_match_toggle") # squad
    addToggles("stats_squads_keeper_for_per_match_toggle") # goalkeeper
    addToggles("stats_squads_keeper_adv_for_per_match_toggle") # advanced keeper
    addToggles("stats_squads_shooting_for_per_match_toggle") # shooting
    addToggles("stats_squads_passing_for_per_match_toggle") # passing
    addToggles("stats_squads_gca_for_per_match_toggle") # goal & shot creation
    addToggles("stats_squads_defense_for_per_match_toggle") # defensive stats
    addToggles("stats_squads_possession_for_per_match_toggle") # possession
    addToggles("stats_squads_playing_time_for_per_match_toggle") # playing time
    
    closeAdButton.click()
    
    for toggle in toggles:
        toggled = False
        while (not toggled): # sometimes the toggle button is not clickable, so we scroll to it and try again
            try:
                toggle.click()
                toggled = True
            except:
                driver.execute_script("arguments[0].scrollIntoView();", toggle) 
        
    html = driver.page_source
    soup = BeautifulSoup(html,'html.parser')
    
    
    # depending on the toggles that are available, the tables are in different indexes
    tableIndexes = [15, 11, 17, 23, 13, 25, 29, 27, 19]
    
    # 11 = squad stats, 13 = keeper stats, 15 = adv goal keeper stats, 17 = squad shooting, 19 = squad passing, 23 = goal and shot creation, 25 = squad defensive actions, 27 = squad possession, 29 = squad playing time
    seperateData = []
    
    for index in tableIndexes: # get the tables from the html\
        try:
            rawData = soup.find_all("table")[index]
            statsTable = pd.read_html(str(rawData)) # returns a list of tables
            statsTable = statsTable[0]
            statsTable.columns = statsTable.columns.droplevel() # removes the multiindex
            seperateData.append(statsTable)
        except:
            pass
    
    driver.close()
    driver.quit()
    
    # merge all the tables into one on the squad column
    data = seperateData[0]
    for i in range(1, len(seperateData)):
        data = data.merge(seperateData[i], on = "Squad")
    
    data = data.loc[:, ~data.columns.duplicated()] # removes duplicate columns
    
    return data

def get_spi_data(url):
    # url leads website with a csv
    # returns a dataframe with the SPI data
    
    response = requests.get(url)
    content = response.content
    
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    
    spi_df = df[['name', 'spi']]
    spi_df = spi_df.rename(columns = {'name': 'Team'})
    
    return spi_df

def cleanOddsJson(league):
    api_key = os.environ.get('THE_ODDS_API_KEY')
    url = f"https://api.the-odds-api.com/v4/sports/{league}/odds/?apiKey={api_key}&regions=us&markets=h2h"
    odds_response = requests.get(url)
    odds_json = odds_response.json()
    simple_json = []
    
    def rename(team):
        if team == "Wolverhampton Wanderers":
            team = "Wolves"
        if team == "Nottingham Forest":
            team = "Nott'ham Forest"
        if team == "West Ham United":
            team = "West Ham"
        if team == "Tottenham Hotspur":
            team = "Tottenham"
        if team == "Newcastle United":
            team = "Newcastle Utd"
        if team == "Brighton and Hove Albion":
            team = "Brighton"
        if team == "Manchester United":
            team = "Manchester Utd"
            
        return team
    
    for game in odds_json: # for each game
        awayID = 1
        homeID = 0
        
        game_id = game['id'] # get the game id
        commence_time = game['commence_time'] # get the game time
        league = game['sport_title'] # get the league
        
        away_team = rename(game['away_team']) # rename the away team
        home_team = rename(game['home_team']) # rename the home team
        
        siteIdx = 0 # index of the bookmaker site
        
        if game['bookmakers'][siteIdx]['markets'][0]['outcomes'][awayID]['name'] != away_team: # if the awayID is not the away team
            awayID = 0 # set the away team to the first outcome
            homeID = 1 # set the home team to the second outcome
        away_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][awayID]['price'] # get the away odds
        home_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][homeID]['price'] # get the home odds
        draw_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][2]['price'] # get the draw odds
        
        while (away_odds == 1.0 or home_odds == 1.0): # if away or home odds are 1.0, then use a different site
            siteIdx += 1 # increment the site index
            if (siteIdx == len(game['bookmakers'])): # if there are no more sites, then skip the game
                break
            if game['bookmakers'][siteIdx]['markets'][0]['outcomes'][awayID]['name'] != away_team: # if the awayID is not the away team
                awayID = 0
                homeID = 1
            away_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][awayID]['price']
            home_odds = game['bookmakers'][siteIdx]['markets'][0]['outcomes'][homeID]['price']

        if (away_odds == 1.0 or home_odds == 1.0): # if away or home odds are 1.0, then skip the game
            continue
        
        # g = [game_id, commence_time, away_team, home_team, away_odds, home_odds, draw_odds] # create a list of the game data
        
        g = {
            'id': game_id,
            'time': commence_time,
            'away_team': away_team,
            'home_team': home_team,
            'away_odds': away_odds,
            'home_odds': home_odds,
            'draw_odds': draw_odds,
            'league': league
        }
        
        simple_json.append(g) # add the game to the list of games
    
    return simple_json

def downloadPastData(seasons):
    past_data = []
    for season in seasons:
        df = downloadData("https://fbref.com/en/comps/9/" + season + "-Premier-League-Stats")
        past_data.append(df)
        
    past_df = pd.concat(past_data)
    
    # save the data to a csv file
    past_df.to_csv("pastData/past_data.csv", index = False)
    
def downloadCurrentData():
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    df = downloadData(url)
    
    # save the data to a csv file
    df.to_csv("currentData/current_data.csv", index = False)
    
def scaleSPI(df):
    # scale the SPI data to be between 0 and 3
    df['spi'] = (df['spi'] - df['spi'].min() + 10) / (df['spi'].max() - df['spi'].min()) * 2 # spi - min(spi) / max(spi) - min(spi) * 3
    
    # cube root the SPI data to make it more linear
    # df['spi'] = df['spi'] ** (1/2)
    
    return df
    
def addSPI(df):
    # get the SPI data
    spi_df = get_spi_data("https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv")
    
    rename_cols = {
        'Manchester United': 'Manchester Utd',
        'Brighton and Hove Albion': 'Brighton',
        'AFC Bournemouth': 'Bournemouth',
        'Newcastle': 'Newcastle Utd',
        'Nottingham Forest': 'Nott\'ham Forest',
        'Tottenham Hotspur': 'Tottenham',
        'West Ham United': 'West Ham',
        'Wolverhampton': 'Wolves',
    }
        
    # rename the teams in the SPI data
    spi_df = spi_df.replace({"Team": rename_cols})
    
    # spi_df is for all teams in the league, so we need to filter it to only include the teams in the current data
    # df has Squad as its index, and spi_df has Team as a column
    # so we need to set the index of spi_df to Team, and then filter it
    spi_df = spi_df.set_index("Team")
    # filter the spi_df to only include the teams in the current data (but remove average row from the index)
    # spi_df = spi_df.loc[df.index.unique().drop("Average")]
    spi_df = spi_df.loc[df.index]
    
    # add the SPI data to the dataframe
    df = df.merge(spi_df, left_index = True, right_index = True)
    
    # scale the SPI data
    df = scaleSPI(df)
    
    return df
    
seasons = ["2021-2022/2021-2022", "2020-2021/2020-2021", "2019-2020/2019-2020", "2018-2019/2018-2019", "2017-2018/2017-2018", "2016-2017/2016-2017", "2015-2016/2015-2016", "2014-2015/2014-2015", "2013-2014/2013-2014", "2012-2013/2012-2013", "2011-2012/2011-2012", "2010-2011/2010-2011", "2009-2010/2009-2010", "2008-2009/2008-2009", "2007-2008/2007-2008", "2006-2007/2006-2007", "2005-2006/2005-2006"]

# downloadCurrentData()
# downloadPastData(seasons[0:5])

# =============================================== [--- Data Pre-Processing ---] ===============================================
def removeDuplicates(df):
    # since i merged the dataframes, there are duplicates (value_x, value_y)
    # rename the columns that end with _x to remove the _x
    df = df.rename(columns = lambda x: x[:-2] if x.endswith('_x') else x)
    
    # remove all duplicate columns keeping the first one
    df = df.loc[:, ~df.columns.duplicated()]

    # remove all columns with _y, _z, or _a since we already have that column
    df = df[df.columns.drop(list(df.filter(regex='_y|_z|_a')))]

    return df

def preprocess_dataframe(df):
    # Remove duplicates and unnecessary columns
    df = removeDuplicates(df)

    # Replace nan values with 0
    df.fillna(0, inplace=True)

    return df

def find_common_columns(df1, df2):
    return list(set(df1.columns).intersection(df2.columns))

# Read the data from the CSV files
current_df = pd.read_csv("currentData/current_data.csv")
past_df = pd.read_csv("pastData/past_data.csv")
teams = current_df["Squad"]

# Preprocess both dataframes
current_df = preprocess_dataframe(current_df)
past_df = preprocess_dataframe(past_df)

# Identify the common columns after preprocessing
common_columns = find_common_columns(current_df, past_df)

# Remove the columns that are not in both dataframes
current_df = current_df[common_columns]
past_df = past_df[common_columns]

past_df.drop(columns = "Squad", inplace = True)
current_df.drop(columns = "Squad", inplace = True)

# normalise the data
scaler = MinMaxScaler()
current_df = pd.DataFrame(scaler.fit_transform(current_df), columns = current_df.columns)
past_df = pd.DataFrame(scaler.fit_transform(past_df), columns = past_df.columns)

# split the data into training and testing data
X_train_gf, X_test_gf, y_train_gf, y_test_gf = train_test_split(past_df.drop(columns='Gls'), past_df["Gls"], test_size = 0.2, random_state = 0)
X_train_ga, X_test_ga, y_train_ga, y_test_ga = train_test_split(past_df.drop(columns='GA'), past_df["GA"], test_size = 0.2, random_state = 0)

# =============================================== [--- Model Training ---] ===============================================
from sklearn.linear_model import Ridge

# train the model
model_gf = Ridge()
model_gf.fit(X_train_gf, y_train_gf)

model_ga = Ridge()
model_ga.fit(X_train_ga, y_train_ga)

# =============================================== [--- Model Testing ---] ===============================================
from sklearn.metrics import mean_squared_error

# test the model
y_pred_gf = model_gf.predict(X_test_gf)
y_pred_ga = model_ga.predict(X_test_ga)

# calculate the mean squared error
mse_gf = mean_squared_error(y_test_gf, y_pred_gf)
mse_ga = mean_squared_error(y_test_ga, y_pred_ga)

print("Score for Goals For: ", model_gf.score(X_test_gf, y_test_gf))
print("Mean Squared Error for Goals For: ", mse_gf)
print("-" * 75)
print("Score for Goals Against: ", model_ga.score(X_test_ga, y_test_ga))
print("Mean Squared Error for Goals Against: ", mse_ga)

# =============================================== [--- Model Prediction ---] ===============================================
def illustrate(df):
    # display a heatmap of the data
    plt.figure(figsize = (12, 8))
    # sort the dataframe by the first column
    sns.heatmap(df.sort_values(by = df.columns[0], ascending = False), annot = True, fmt = ".2f", cmap = "Blues")
    plt.show()
    
def calculate_strength(expected, average):
    return expected / average

def calculate_off_strength(expected, average, spi):
    return (expected / average) * (spi / 3)

def calculate_def_strength(expected, average, spi):
    return (expected / average) * (spi / 1.5)

def close_gap(df, column, limit, min_value):
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    # Calculate the range of the original values
    original_range = df[column].max() - df[column].min()

    # Check if the original range is 0 to avoid division by zero
    if original_range == 0:
        raise ValueError("The original range of values is 0, cannot rescale.")

    # Calculate the scale factor
    scale_factor = limit / original_range

    # Calculate the new min value
    new_min = df[column].min()

    # Apply the transformation
    df[column] = (df[column] - new_min) * scale_factor
    
    # Check if the new min value is less than the minimum value
    if df[column].min() < min_value:
        # Calculate the difference
        diff = min_value - df[column].min()
        
        # Add the difference to all the values
        df[column] = df[column] + diff

    return df

prediction_df = pd.DataFrame(columns = ["Predicted Gls", "Predicted GA"], index = teams)

prediction_df["Predicted Gls"] = model_gf.predict(current_df.drop(columns = "Gls"))
prediction_df["Predicted GA"] = model_ga.predict(current_df.drop(columns = "GA"))

# add the absolute minimum value to all the values in the dataframe
prediction_df["Predicted Gls"] = prediction_df["Predicted Gls"] + abs(prediction_df["Predicted Gls"].min())
prediction_df["Predicted GA"] = prediction_df["Predicted GA"] + abs(prediction_df["Predicted GA"].min())

prediction_df = addSPI(prediction_df)

# add an average row at the bottom for each column
prediction_df.loc['Average'] = prediction_df.mean()

strength_df = pd.DataFrame(columns = ["Offensive Strength", "Defensive Strength"], index = prediction_df.index)

# calculate the offensive strength of each team
# strength_df["Offensive Strength"] = prediction_df.apply(lambda x: calculate_strength(x["Predicted Gls"], prediction_df.loc["Average", "Predicted Gls"]), axis = 1)
# strength_df["Defensive Strength"] = prediction_df.apply(lambda x: calculate_strength(x["Predicted GA"], prediction_df.loc["Average", "Predicted GA"]), axis = 1)

strength_df["Offensive Strength"] = prediction_df.apply(lambda x: calculate_off_strength(x["Predicted Gls"], prediction_df.loc["Average", "Predicted Gls"], x["spi"]), axis = 1)
strength_df["Defensive Strength"] = prediction_df.apply(lambda x: calculate_def_strength(x["Predicted GA"], prediction_df.loc["Average", "Predicted GA"], x["spi"]), axis = 1)

# normalize offensive strength
strength_df = close_gap(strength_df, "Offensive Strength", .5, .5)
strength_df = close_gap(strength_df, "Defensive Strength", .5, .5)

# illustrate(strength_df)

# =============================================== [--- Predict Games ---] ===============================================
from scipy.stats import poisson

# using the strengths of each team, predict the outcome of each game
def predictGames(awayTeam, homeTeam, strengths, predictions):
    # get the strengths of each team
    home_attack_strength = strengths.loc[homeTeam]['Offensive Strength']
    home_defense_strength = strengths.loc[homeTeam]['Defensive Strength']
    
    away_attack_strength = strengths.loc[awayTeam]['Offensive Strength']
    away_defense_strength = strengths.loc[awayTeam]['Defensive Strength']
    
    # calculate the expected goals for and goals against
    home_expected_gf = home_attack_strength * away_defense_strength * predictions.loc['Average']['Predicted Gls']
    
    away_expected_gf = away_attack_strength * home_defense_strength * predictions.loc['Average']['Predicted Gls']
    
    awayProb = 0
    homeProb = 0
    tieProb = 0
    
    for i in range(0, 10):
        for j in range(0, 10):
            # calculate the probability of each outcome
            prob = poisson.pmf(i, away_expected_gf) * poisson.pmf(j, home_expected_gf)
            
            if i > j:
                awayProb += prob
            elif i < j:
                homeProb += prob
            else:
                tieProb += prob
                
    awayProb = awayProb + (tieProb / 2)
    homeProb = homeProb + (tieProb / 2)
                
    return homeProb, awayProb

# =============================================== [--- Odds ---] ===============================================
def decimalOddsToPercent(odds):
    return (1 / odds) * 100

def getPredictionsFromOdds(simple_json):
    predictions = []
    for game in simple_json:
        homeTeam = game["home_team"]
        awayTeam = game["away_team"]
        
        homeProb, awayProb = predictGames(awayTeam, homeTeam, strength_df, prediction_df)
        
        # copy the game dictionary and add the probabilities
        pred = game.copy()
        pred["home_prob"] = homeProb * 100
        pred["away_prob"] = awayProb * 100
    
        predictions.append(pred)
        
    return predictions

def addEdgeToPredictions(predictions):
    for game in predictions:
        homeProb = game["home_prob"]
        awayProb = game["away_prob"]
        
        vegasHomeProb = decimalOddsToPercent(game["home_odds"])
        vegasAwayProb = decimalOddsToPercent(game["away_odds"])
        
        game["vegas_home_prob"] = vegasHomeProb
        game["vegas_away_prob"] = vegasAwayProb
                
        # calculate the edge we have over the bookies
        game["home_edge"] = homeProb - vegasHomeProb
        game["away_edge"] = awayProb - vegasAwayProb
        
    return predictions

def filterPredictionsByEdge(predictions, minEdge):
    filtered = []
    for game in predictions:
        if game["home_edge"] > minEdge or game["away_edge"] > minEdge:
            filtered.append(game)
            
    return filtered

simple_json = cleanOddsJson("soccer_epl")
simple_json = getPredictionsFromOdds(simple_json)
simple_json = addEdgeToPredictions(simple_json)
filtered_picks = filterPredictionsByEdge(simple_json, 5)

# turn the list of dictionaries into a dataframe
simple_df = pd.DataFrame(simple_json)

print(simple_df)

# =============================================== [--- Database (firebase real-time) ---] ===============================================
def initFirebaseDatabase():
    cred = credentials.Certificate('functions/algopick-acc98-firebase-adminsdk-mwc8r-f53eed233a.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://algopick-acc98-default-rtdb.firebaseio.com'
    })
    print("Database Initialized: " + str(db.reference().get() != None))

initFirebaseDatabase()

def getPredictionsFromDatabase():
    ref = db.reference()
    return ref.child("predictions").get()

def samePrediction(prediction1, prediction2):
    return prediction1["home_team"] == prediction2["home_team"] and prediction1["away_team"] == prediction2["away_team"] and prediction1['time'] == prediction2['time']

def addPredictionsToDatabase(predictions):
    ref = db.reference()
    
    # get the current predictions
    current_predictions = getPredictionsFromDatabase()
    
    # if there are no predictions, add the new ones
    if current_predictions == None:
        ref.child("predictions").set(predictions)
    else:
        # if there are predictions, add the new ones to the end
        # add the new predictions to the end of the list making sure it is unique
        for prediction in predictions:
            if not any(samePrediction(prediction, pred) for pred in current_predictions):
                current_predictions.append(prediction)
        
        ref.child("predictions").set(current_predictions)
    
    print("Predictions added: " + str(ref.child("predictions").get() != None))
    
addPredictionsToDatabase(simple_json)

print("Code completed.")