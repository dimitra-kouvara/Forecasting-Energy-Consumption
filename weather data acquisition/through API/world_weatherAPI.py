"""
PROJECT: FORECASTING RESIDENTIAL CONSUMPTION
Weather Scraping through API

Site - weather operator
https://www.worldweatheronline.com/

@author: Dimitra
"""

# Import the necessary libraries
# pip install WorldWeatherPy
from WorldWeatherPy import DetermineListOfAttributes
from WorldWeatherPy import HistoricalLocationWeather
from WorldWeatherPy import RetrieveByAttribute
import pandas as pd
import weather_configuration

# the API key obtained from https://www.worldweatheronline.com/developer/. (str)
api_key = weather_configuration.api_key
# a city for which to retrieve data. (str).
city = weather_configuration.city
# a string in the format YYYY-MM-DD (str).
start_date = weather_configuration.start_date
# a string in the format YYYY-MM-DD (str).
end_date = weather_configuration.end_date
# the frequency of extracted data, measured in hours. (int)
frequency = weather_configuration.frequency

dataset = HistoricalLocationWeather(api_key, city, start_date, end_date, frequency).retrieve_hist_data()
# Returns a Pandas DataFrame 'dataset', which contains an array of weather attributes for the given city, 
# between the start and end dates specified, with hourly frequency, indexed by date and time.

# Save data to csv file
dataset.to_csv(city + '_hourly_weather.csv')
