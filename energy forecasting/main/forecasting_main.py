import math
import re
import json
import pandas as pd
import numpy as np
import datetime
import time
import holidays
import itertools
import random
import os
import statistics
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import keras
from scikeras.wrappers import KerasRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from fbprophet import Prophet
from sklearn.metrics import r2_score
from keras.layers import LSTM, Dense

# Start a timer to count the time it needs to run the model
start = time.time()

# Read pipeline configuration as json and save the parameters to variables
pipeline = json.load(open('pipeline.config'))

high_correlated_features = pipeline['forecaster_main']['high_correlated_features']
show_plots = pipeline['forecaster_main']['show_plots']
save_plots = pipeline['forecaster_main']['save_plots']
use_only_correct_data = pipeline['forecaster_main']['use_only_correct_data']
fix_if_sensor_restart = pipeline['forecaster_main']['fix_if_sensor_restart']
save_predictions = pipeline['forecaster_main']['save_predictions']
choose_model_to_run = pipeline['forecaster_main']['choose_model_to_run']
save_evaluation = pipeline['forecaster_main']['save_evaluation']
good_time_start = pipeline['forecaster_main']['good_time_start']
good_time_end = pipeline['forecaster_main']['good_time_end']
path_to_energy_data = pipeline['forecaster_main']['path_to_energy_data']
path_to_power_data = pipeline['forecaster_main']['path_to_power_data']
path_to_temp_humi_data = pipeline['forecaster_main']['path_to_temp_humi_data']
path_to_weather_data = pipeline['forecaster_main']['path_to_weather_data']
scaler = pipeline['forecaster_main']['scaler']
site_number = re.findall('(\w{4}\d+)_.+', string=path_to_energy_data)[0]

regression_test_size = pipeline['regression']['test_size']
regression_only_energy = pipeline['regression']['only_energy']

rnn_test_size = pipeline['recurrent_neural_network']['test_size']
rnn_only_energy = pipeline['recurrent_neural_network']['only_energy']
rnn_batch_size = pipeline['recurrent_neural_network']['batch_size']
rnn_epochs = pipeline['recurrent_neural_network']['epochs']
rnn_optimizer = pipeline['recurrent_neural_network']['optimizer']

arima_test_size = pipeline['arima']['test_size']
arima_only_energy = pipeline['arima']['only_energy']

sarimax_test_size = pipeline['sarimax']['test_size']
sarimax_only_energy = pipeline['sarimax']['only_energy']
exogeneous_data = pipeline['sarimax']['exogeneous_data']
grid_search_for_order_params = pipeline['sarimax']['grid_search_for_order_params']

prophet_test_size = pipeline['fb_prophet']['test_size']
prophet_only_energy = pipeline['fb_prophet']['only_energy']
add_holidays = pipeline['fb_prophet']['add_holidays']
period = pipeline['fb_prophet']['period']
fourier_order = pipeline['fb_prophet']['fourier_order']


class Forecaster():
    '''
    This class is responsible for the whole process. Reads the files, preprocess them and gives a prediction.
    Takes as input the parameters from the pipeline configuration file.
    '''

    def __init__(self, high_correlated_features, energy_json_directory, power_json_directory, temp_humi_json_directory,
                 weather_csv_directory, show_plots, save_plots, use_only_correct_data, fix_if_sensor_restart, save_predictions, save_evaluation,
                 good_time_start, good_time_end, start_time, scaler_f, site_number):
        self.high_correlated_features = high_correlated_features
        self.energy_json_directory = energy_json_directory
        self.power_json_directory = power_json_directory
        self.temp_humi_json_directory = temp_humi_json_directory
        self.weather_csv_directory = weather_csv_directory
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.use_only_correct_data = use_only_correct_data
        self.fix_if_sensor_restart = fix_if_sensor_restart
        self.save_predictions = save_predictions
        self.save_evaluation = save_evaluation
        self.good_time_start = good_time_start
        self.good_time_end = good_time_end
        self.start_time = start_time
        self.scaler_f = scaler_f
        self.site_number = site_number

        # make a folder for the site you run
        if self.save_plots:
            cwd = os.getcwd()
            new_folder = self.site_number
            site_directory = os.path.join(cwd, new_folder)
            isExist = os.path.exists(site_directory)
            if not isExist:
                os.makedirs(site_directory)
            self.site_directory = site_directory
        
        # make a folder to save the plots
        if self.save_plots:
            cwd = self.site_directory
            new_folder = 'plots'
            plot_directory = os.path.join(cwd, new_folder)
            isExist = os.path.exists(plot_directory)
            if not isExist:
                os.makedirs(plot_directory)
            self.plot_directory = plot_directory

        # make a folder to save the predictions in csv
        if self.save_predictions:
            cwd = self.site_directory
            new_folder = 'predictions'
            predictions_directory = os.path.join(cwd, new_folder)
            isExist = os.path.exists(predictions_directory)
            if not isExist:
                os.makedirs(predictions_directory)
            self.predictions_directory = predictions_directory
        
        # make a folder to save the evaluation in csv
        if self.save_evaluation:
            cwd = self.site_directory
            new_folder = 'evaluation'
            evaluation_directory = os.path.join(cwd, new_folder)
            isExist = os.path.exists(evaluation_directory)
            if not isExist:
                os.makedirs(evaluation_directory)
            self.evaluation_directory = evaluation_directory

    # Function that takes as input the directory where the json files reside and returns them in a dataframe
    def load_data_from_database(self, name_of_file):
        def read_json(jsonName):
            with open(jsonName) as outfile:
                # returns JSON object as dictionary
                sensor_data = json.load(outfile)
                outfile.close()

            return sensor_data

        # Loading the data from json file
        sensor_raw_data = read_json(jsonName=name_of_file)
        sensor_names = sensor_raw_data['results'][0]['series'][0]['columns']
        sensor_data = sensor_raw_data['results'][0]['series'][0]['values']
        no_of_names = len(sensor_names)

        # Converting the data to DataFrames
        sensor_df = pd.DataFrame(sensor_data, columns=[sensor_names[i] for i in range(no_of_names)])

        return sensor_df

    # Function that takes as input the directory where the csv files reside and returns them in a dataframe
    def load_data_from_csv(self, name_of_file):
        df = pd.read_csv(name_of_file)

        return df

    # Function that creates extra time-related features from weather data (sunset, sunrise, isdaylight)
    def feature_extraction_weather_data(self):
        # Function that converts the format of the timestamp
        def convert_pmANDam(hour):
            try:
                in_time = datetime.datetime.strptime(hour, "%I:%M %p")
                out_time = datetime.datetime.strftime(in_time, "%H:%M")
            except:
                out_time = None
            return out_time

        df_weather = self.load_data_from_csv(self.weather_csv_directory)

        print(f'Weather data Descriptive statistics and missing values:')
        print(df_weather.describe())
        percent_missing = df_weather.isnull().sum() * 100 / len(df_weather)
        missing_value_df = pd.DataFrame({'column_name': df_weather.columns,
                                 'percent_missing': percent_missing})
        print(missing_value_df, '\n')
        print('--------------------------------------------------------\n')

        df_weather.rename(columns={'date_time': 'timestamp', 'humidity': 'outdoorHumidity'}, inplace=True)
        df_weather['timestamp'] = df_weather['timestamp'].astype('datetime64[ns]')
        df_weather['time'] = df_weather['timestamp'].dt.time
        # Converting time to another format
        df_weather['sunrise'] = df_weather['sunrise'].apply(lambda x: convert_pmANDam(x))
        df_weather['sunset'] = df_weather['sunset'].apply(lambda x: convert_pmANDam(x))

        df_weather['sunrise'] = pd.to_datetime(df_weather['sunrise'], format='%H:%M').dt.time
        df_weather['sunset'] = pd.to_datetime(df_weather['sunset'], format='%H:%M').dt.time

        df_weather['isdaylight'] = np.where(
            (df_weather['time'] >= df_weather['sunrise']) & (df_weather['time'] <= df_weather['sunset']), 1, 0)

        # Deleting unnecessary columns
        df_weather.drop(columns=['moonrise', 'moonset', 'time'], inplace=True)

        return df_weather
    

    # Function that when initiated loads the dataframes of each json file , makes some descriptive statistics and extracts features from the data
    def feature_extraction_sensor_data(self):
        # Function that preserves only the English version of the bank holiday
        def engVersion_holidays(text):
            return text[text.find("[") + 1:text.find("]")]

        # Function that erases special characters 
        def no_special(text):
            return re.sub("[^a-zA-Z0-9]+", "", text)

        # load the energy data 
        energy_df = self.load_data_from_database(self.energy_json_directory)
        # print('Energy data have: ', energy_df.shape[0], ' rows')

        # Getting the dataframe's descriptive statistics
        print('--------------------------------------------------------')
        print('Descriptive statistics for the dataset')
        print(energy_df.describe())

        # percentage of missing values for each column of the fataframe
        percent_missing = energy_df.isnull().sum() * 100 / len(energy_df)
        missing_value_df = pd.DataFrame({'column_name': energy_df.columns,
                                 'percent_missing': percent_missing})
        print(missing_value_df)

        plt.figure()
        plt.title('Energy total before data cleaning')
        plt.xlabel('Time [hours]')
        plt.ylabel('Energy [kWh]')
        plt.plot(energy_df['energy_total'])

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'EnergyTotalBeforeDataCleaning.png'))

        if self.fix_if_sensor_restart == True:
            energy_df = self.fix_if_sensor_restarted(energy_df)

        plt.figure()
        plt.title('Energy total after fix if sensor restart')
        plt.xlabel('Time [hours]')
        plt.ylabel('Energy [kWh]')
        plt.plot(energy_df['energy_total'])

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'EnergyTotalAfterFixIfSensorRestarted.png'))

        # load data for power
        power_df = self.load_data_from_database(self.power_json_directory)
        # print('Power data have: ', power_df.shape[0], ' rows')

        # power data are in kilowatt for this house so for concistency and visualization purposes, they are transformed to watts 
        power_df['power'] = power_df['power'].apply(lambda x: x*1000)

        # Getting the dataframe's descriptive statistics
        print(power_df.describe())

        percent_missing = power_df.isnull().sum() * 100 / len(power_df)
        missing_value_df = pd.DataFrame({'column_name': power_df.columns,
                                 'percent_missing': percent_missing})
        print(missing_value_df)

        # this df will have only energy values because the temperature data are less (around 5 months) and so we have more energy values
        df_only_energy = pd.merge(energy_df, power_df, on='time', how='left')
        # print('Only energy has: ', df_only_energy.shape[0], ' rows\n')

        # load data for temperature and humidity indoor sensors
        temp_humi_df = self.load_data_from_database(self.temp_humi_json_directory)
        # print('Temperature data had: ', temp_humi_df.shape[0], ' rows')

        # Getting the dataframe's descriptive statistics
        print(temp_humi_df.describe())
        percent_missing = temp_humi_df.isnull().sum() * 100 / len(temp_humi_df)
        missing_value_df = pd.DataFrame({'column_name': temp_humi_df.columns,
                                 'percent_missing': percent_missing})
        print(missing_value_df)

        plt.figure()
        plt.title('Temperature before data cleaning')
        plt.xlabel('Time [hours]')
        plt.ylabel('Temperature [C]')
        plt.plot(temp_humi_df['temperature'])

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'TemperatureDataBeforeCleaning.png'))
        
        plt.figure()
        plt.title('Humidity before data cleaning')
        plt.xlabel('Time [hours]')
        plt.ylabel('Humidity [%]')
        plt.plot(temp_humi_df['humidity'])

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'HumidityDataBeforeCleaning.png'))

        # df_sensor id the dataframe that has data from all json files merged (only the intersection data)
        df_sensor = pd.merge(energy_df, temp_humi_df, on='time', how='outer')
        # print('Merged data had: ', df_sensor.shape[0], ' rows')

        df_sensor = pd.merge(df_sensor, power_df, on='time', how='outer')
        # print('Merged data had: ', df_sensor.shape[0], ' rows\n')

        # Here only the "good" data for each house are extracted. We have manually picked the region before the sensor restart 
        if self.use_only_correct_data == True:
            df_sensor = df_sensor[df_sensor['time']>=self.good_time_start]
            df_sensor = df_sensor[df_sensor['time']<self.good_time_end]
            df_only_energy = df_only_energy[df_only_energy['time']>=self.good_time_start]
            df_only_energy = df_only_energy[df_only_energy['time']<self.good_time_end]
            df_sensor = df_sensor.reset_index()
            df_only_energy = df_only_energy.reset_index()

        # creation of timestamp (data type: datetime64)
        df_sensor = df_sensor.rename(columns={'time': 'timestamp'})
        # To remove timezone, use tz_localize:
        df_sensor['timestamp'] = pd.to_datetime(df_sensor.timestamp).dt.tz_localize(None)

        # creation of timestamp (data type: datetime64)
        df_only_energy = df_only_energy.rename(columns={'time': 'timestamp'})
        # To remove timezone, use tz_localize:
        df_only_energy['timestamp'] = pd.to_datetime(df_only_energy.timestamp).dt.tz_localize(None)

        df_only_energy['date'] = df_only_energy['timestamp'].dt.date
        df_only_energy['year'] = df_only_energy['timestamp'].dt.year

        # creation of year
        df_sensor['year'] = df_sensor['timestamp'].dt.year
        # creation of month (Jan: 1 to Dec: 12)
        df_sensor['month'] = df_sensor['timestamp'].dt.month
        # creation of date (form: 2021-09-08)
        df_sensor['date'] = df_sensor['timestamp'].dt.date
        # creation of day
        df_sensor['day'] = df_sensor['timestamp'].dt.day
        # creation of day of week (Mon: 0 to Sun: 6)
        df_sensor['day_of_week'] = df_sensor['timestamp'].dt.dayofweek
        # creation of time (form: 08:09:30)
        df_sensor['time'] = df_sensor['timestamp'].dt.time
        # creation of hour
        df_sensor['hour'] = df_sensor['timestamp'].dt.hour
        # weekday 1 or weekend 0 (if the day is less than 5 append 1 to column weekday, else append 0)
        df_sensor['weekday'] = np.where(df_sensor['day_of_week'].lt(5), 1,
                                        0)  # df_sensor['timestamp'].dt.dayofweek.lt(5), 1, 0)
        # working time from 9:00am until 18:00pm: 1 or resting time: 0
        workingHours = [(df_sensor['timestamp'].dt.hour.le(18)) &
                        df_sensor['timestamp'].dt.dayofweek.lt(5) &
                        (df_sensor['timestamp'].dt.hour.ge(9))]
        df_sensor['workingHours'] = np.select(workingHours, '1', default='0')
        # busy time: 7:00 - 9:00 on weekdays, 9:00 - 15:00 on weekends & 19:00 - 00:30 during the week: 1 or resting time: 0
        busyHours = [((df_sensor['timestamp'].dt.hour.ge(7)) &
                      (df_sensor['timestamp'].dt.hour.le(9)) &
                      df_sensor['timestamp'].dt.dayofweek.lt(5)) |
                     ((df_sensor['timestamp'].dt.hour.ge(9)) &
                      (df_sensor['timestamp'].dt.hour.le(15)) &
                      df_sensor['timestamp'].dt.dayofweek.isin([5, 6])) |
                     ((df_sensor['timestamp'].dt.hour.ge(19)) &
                      (df_sensor['timestamp'].dt.hour.le(24)))]
        df_sensor['busyHours'] = np.select(busyHours, '1', default='0')

        # Computing holidays
        gr_holidays = holidays.Greece()

        # df_sensor['datetime'] = pd.to_datetime(df_sensor['day'])
        df_sensor['datetime'] = df_sensor['date']
        df_sensor['holiday'] = pd.Series(df_sensor['datetime']).apply(lambda x: gr_holidays.get(x)).values
        # Addition of bank holidays for the fbprophet model
        gr_public_holidays = df_sensor[['day', 'holiday']]
        gr_public_holidays = gr_public_holidays.replace(to_replace='None', value=np.nan).dropna()
        gr_public_holidays.drop_duplicates(subset=None, keep='first', inplace=True)
        gr_public_holidays['holiday'] = gr_public_holidays['holiday'].apply(engVersion_holidays)

        df_sensor['holiday'].replace({None: 'NoHoliday'}, inplace=True)
        # Getting one hot encoding of columns holiday
        one_hot = pd.get_dummies(df_sensor['holiday'])
        # Dropping column holiday as it is now encoded
        df_sensor = df_sensor.drop('holiday', axis=1)
        # Joining the encoded df
        df_sensor = df_sensor.join(one_hot)
        df_sensor = df_sensor.rename(columns=lambda x: no_special(x))

        df_only_energy = df_only_energy.rename(columns=lambda x: no_special(x))
        

        return df_sensor, df_only_energy
    
    # This function is responsible to identify the datapoints where the sensor gave steady energy values
    def identify_sensor_faults(self, di):
        # Forward fill for the very small percentage of missing values
        # This forward fill will (fill values with the next valid value) does not affect the data as it will create
        # steady energy consumption which the function will fix
        di.fillna(method='ffill', inplace=True)

        # parameter that stores how many times the same value exists
        times_same_val = 0
        # parameter for the value of the previous iteration
        last_value = 0
        # a list to save the indexes of the "bad" energy values
        index_of_measurements = []
        # a dictionary to store all the lists (index_of_measurements)
        dict_of_indexes = {}
        # counter for the number of lists that will be made
        count = 0
        # for how many concecutive steady energy values to look
        concecutive_steady_power = 2
        # the minimum energy consumption to flaf a value as steady (if next_value - previous_value < 0.06 consider it as steady)
        minimum_energy_consumed_in_one_hour = 0.06

        # iterate through all the energy values and their corresponding index
        for i, idx in zip(di["energytotal"], di.index.values.tolist()):
            # if the current value minus the value from the previous iteration is smaller than the minimum (0.06) consider it steady
            if np.abs(round(i, 4) - round(last_value, 4)) < minimum_energy_consumed_in_one_hour:
                # if it is the first time it is being steady, put also the previous index (previous value)
                if times_same_val == 0:
                    index_of_measurements.append(idx - 1)
                
                # add 1 to the number of times it sees the same value
                times_same_val += 1
                # append the index of the current energy value to the list of indexes
                index_of_measurements.append(idx)

                # if it is not the last row of the dataframe, then locate the next energy value
                if idx != len(di.index.values.tolist())-1:
                    next_value = di.energytotal.iloc[idx + 1]
                
                # if the times_same_val parameter is bigger than the threshold for concecutive times (here is 2) and the difference between current and next value is
                # bigger than 0.06 (meaning not steady), then store the list of indexes and re-initiate all the parameters
                if times_same_val > concecutive_steady_power and np.abs(round(i, 4) - round(next_value, 4)) >= minimum_energy_consumed_in_one_hour:
                    # print("check",last_value,i,next_value)
                    dict_of_indexes[str(count)] = index_of_measurements
                    count += 1
                    index_of_measurements = []
                    times_same_val = 0
            
            # else, reinitiate all the parameters
            else:
                times_same_val = 0
                index_of_measurements = []
            
            # the last value is stored as the current value (on the beggining of the next iteration, this will be the previious value)
            last_value = i

        # this for loop adds some values before and after the steady parts to depict them better
        fig_counter = 0
        for key in dict_of_indexes:
            list_of_meas = dict_of_indexes[key]
            
            first_idx = list_of_meas[0]
            if fig_counter == len(dict_of_indexes) - 1:
                show_x_before_and_after_values = 0
                continue
            else:
                show_x_before_and_after_values = 5

            for i in range(1, show_x_before_and_after_values+1):
                list_of_meas.insert(0, first_idx - i)

            last_idx = list_of_meas[-1]
            for i in range(1, show_x_before_and_after_values+1):
                list_of_meas.append(last_idx + i)

            # plot the parts where power is steady
            plt.figure()
            plt.title(f'Power is steady for {len(dict_of_indexes[key])} consecutive times')
            plt.plot(di["power"][list_of_meas])

            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory, 'PowerFailure'+str(fig_counter)+'.png'))
            
            # plot the parts where energy is steady
            plt.figure()
            plt.title(f'Energy is steady for {len(dict_of_indexes[key])} consecutive times')
            plt.plot(di["energytotal"][list_of_meas])

            if self.save_plots == True:
                time.sleep(0.5)
                plt.savefig(os.path.join(self.plot_directory, 'EnergyFailure'+str(fig_counter)+'.png'))
            
            fig_counter += 1
            
            # from the indexes discovored, put the energy prices as zero, in order to fix them in the cleaning procedure
            for idx in dict_of_indexes[key]:
                di.energytotal.iloc[idx] = 0
        
        return di
    
    # Function that checks if the meter had a restart malfunction and tries to fix it
    # This function can only be applied to site22 and basically doea not work properly (it returns)
    def fix_if_sensor_restarted(self, energy_df):
        try:
            # a list of lists containing energy values and the corresponding timestamp
            data = energy_df[['time', 'energy_total']].values.tolist()
            # a parameter to store the last valid value
            last_seen_valid_value = 0

            for numerator in range(len(data)):
                # parameter for the current value
                current_value = data[numerator][1]
                # parameter to count the concecutive times where the next value is wrong (smaller than the previous)
                counter_consec_times = 0
                # a list to store the wrong values
                list_of_concec_wrong = []

                # if the currnet value is not nan 
                if not math.isnan(current_value):
                    # check to see if the next values is smaller than the previous
                    if current_value < last_seen_valid_value:
                        # if it is, then iterate through the next value and afterwards
                        for values in data[numerator + 1:]:
                            # add one to the counter (found wrong value)
                            counter_consec_times += 1
                            # specify the energy value (0 index is timestamp, 1 index is the energy value)
                            next_seen_valid_value = values[1]
                            # add the parameter to the list with wrong parameters
                            list_of_concec_wrong.append(next_seen_valid_value)
                            # if the next value in not None and is bigger than the last good value
                            if next_seen_valid_value is not None and next_seen_valid_value > last_seen_valid_value:
                                # and if the counter is less than 2000 times (manually inserted after watching the faulty period), break the loop (stop searching for the current interval)
                                if counter_consec_times <= 2000:
                                    break

                                # else, if it is more than 2000 that means that the sensor has restarted
                                else:
                                    # the last valid value should be counter_consec_times before but it is not, so we manually inserted 2083 through trial and error
                                    last_seen_valid_value = energy_df['energy_total'][numerator - 2083]

                                    # print('Energy counter has restarted')
                                    # the energy value afterwards should be the first (meaning restarted was made so it went to 0 or another number)
                                    # but it is not because the sensor had another malfunction, so we tried to approximate it with the median value
                                    energy_val_of_meter_after_restart = statistics.median(list_of_concec_wrong)

                                    # here is a for loop that adds the difference in order bring the energy values to the correct
                                    # but it is not working due to the problems described above
                                    # for ind in energy_df.index:
                                    #     if ind>=numerator-30:
                                            # energy_df['energy_total'][ind] = energy_df['energy_total'][ind] + (last_seen_valid_value - energy_val_of_meter_after_restart)

                                    # here we drop data that could not be fixed (data where the sensor restart and afterwards)
                                    energy_df = energy_df.drop(range(15500, len(data)), axis=0)
                                    
                                    return energy_df

                        # here we fix the times where the sensor did not restart by taking the mean of the previous valid value and the next valid value
                        current_new_value = np.mean([last_seen_valid_value, next_seen_valid_value])
                        energy_df.loc[numerator, 'energy_total'] = current_new_value
                        current_value = current_new_value

                # else if it is nan skip the current value and try to find the next valid value (from the next value and afterwards)
                # and do the same process as above
                else:
                    for values in data[numerator + 1:]:
                        next_seen_valid_value = values[1]
                        if next_seen_valid_value is not None and next_seen_valid_value > last_seen_valid_value:
                            if counter_consec_times <= 2000:
                                break
                            else:
                                exit()
                                
                    current_new_value = np.mean([last_seen_valid_value, next_seen_valid_value])
                    energy_df.loc[numerator, 'energy_total'] = current_new_value
                    current_value = current_new_value

                if current_value is not None:
                    last_seen_valid_value = current_value
        
        except:
            return energy_df

    
    def preprocessing(self):
        # make timestamp to unix
        def ts_to_unix(s):
            return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())

        # A function that fixes the the remained energy values and computes the energy difference per hour
        def fix_and_compute_energy_difference(df, only_energy):
            # the last valid value that was captured
            last_seen_valid_value = 0

            # remove outliers for temperature humidity and energy via zscore
            if only_energy == False:
                df['zscore_temp'] = np.abs((df['temperature'] - df['temperature'].mean()) / df['temperature'].std(ddof=0))
                df['temperature'] = df['temperature'].mask(df['zscore_temp'] > 3)  # If filter by masking.

                df['zscore_humi'] = np.abs((df['humidity'] - df['humidity'].mean()) / df['humidity'].std(ddof=0))
                df['humidity'] = df['humidity'].mask(df['zscore_humi'] > 3)  # If filter by masking.

            df['zscore_energy'] = np.abs((df['energytotal'] - df['energytotal'].mean()) / df['energytotal'].std(ddof=0))
            df['energytotal'] = df['energytotal'].mask(df['zscore_energy'] > 3)  # If filter by masking.

            # a list of lists containing the energy values and the corresponding timestamp
            data = df[['timestamp', 'energytotal']].values.tolist()

            # Iterate through datapoints
            for numerator in range(len(data)):
                # specify the energy value (0 index is timestamp, 1 index is the energy value)
                current_value = data[numerator][1]
                
                # if the value is not nan
                if not math.isnan(current_value):
                    # and if the value is smaller thanm the last (previous) value
                    if current_value < last_seen_valid_value:
                        # iterate on the next values until you find a value that is bigger than the last seen valid
                        for values in data[numerator + 1:]:
                            next_seen_valid_value = values[1]
                            if next_seen_valid_value is not None and next_seen_valid_value > last_seen_valid_value:
                                break

                        # the current value will be the mean of the last seen and the next valid value
                        current_new_value = np.mean([last_seen_valid_value, next_seen_valid_value])

                        # update also the dataframe
                        df.loc[numerator, 'energytotal'] = current_new_value
                        # the current value becomes the updated one
                        current_value = current_new_value

                # else, if the value is nan, iterate through the next values and when you find a valid one follow the same procedure
                else:
                    for values in data[numerator + 1:]:
                        next_seen_valid_value = values[1]
                        if next_seen_valid_value is not None and next_seen_valid_value > last_seen_valid_value:
                            break
                    current_new_value = np.mean([last_seen_valid_value, next_seen_valid_value])
                    df.loc[numerator, 'energytotal'] = current_new_value
                    current_value = current_new_value

                # the last seen valid is the updated current value (on next iteration will be the previous value)
                if current_value is not None:
                    last_seen_valid_value = current_value

            # compute the energy difference (energy consumed during the hour)
            df['Energy_Difference'] = df['energytotal'].diff()

            plt.figure()
            plt.title('Energy Difference before data cleaning')
            plt.xlabel('Time [hours]')
            plt.ylabel('Energy [kWh]')  ## Total Energy Consumption
            plt.plot(df['Energy_Difference'])

            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory ,'EnergyDifferenceBeforeCleaning.png'))

            # Replace the first value with the mean
            df.loc[0, 'Energy_Difference'] = df['Energy_Difference'].mean()

            # how many values are less than 0.06
            # print('Before cleaning under 0.06 values: ',(df['Energy_Difference']<=0.06).sum())

            # Zscore to delete the outliers
            df['zscore_energyDifference'] = np.abs(
                (df['Energy_Difference'] - df['Energy_Difference'].mean()) / df['Energy_Difference'].std(ddof=0))
            df['Energy_Difference'] = df['Energy_Difference'].mask(
                df['zscore_energyDifference'] > 3)  # If filter by masking.
            
            # when the energy consumption is lower than 0.06 in an hour, delete it
            # 0.06 is the minimum consumption that a house can have in an hour
            df['Energy_Difference'] = np.where(df['Energy_Difference'] <= 0.06, np.nan, df['Energy_Difference'])

            # Fill the deleted values with the next valed value
            df.fillna(method='ffill', inplace=True)

            plt.figure()
            plt.title('Energy Difference after data cleaning')
            plt.xlabel('Time [hours]')
            plt.ylabel('Energy [kWh]')  ## Total Energy Consumption
            plt.plot(df['Energy_Difference'])
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory ,'EnergyDifferenceAfterCleaning.png'))

            plt.figure()
            plt.title('Energy total after data cleaning')
            plt.xlabel('Time [hours]')
            plt.ylabel('Energy [kWh]')  ## Total Energy Consumption
            plt.plot(df['energytotal'])
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory ,'EnergyTotalAfterCleaning.png'))

            return df

        # use the previous functions to get the dataframes in order to preprocess them
        sensor_data, energy_data = self.feature_extraction_sensor_data()
        weather_data = self.feature_extraction_weather_data()

        # merge the data from the sensors with the data from weather API
        df = pd.merge(sensor_data, weather_data, on='timestamp')

        # convert sunrise and sunset features to string format
        df['sunrise'] = df['date'].astype(str) + ' ' + df['sunrise'].astype(str)
        df['sunset'] = df['date'].astype(str) + ' ' + df['sunset'].astype(str)

        df['sunrise'] = df['sunrise'].apply(ts_to_unix)
        df['sunset'] = df['sunset'].apply(ts_to_unix)

        df['workingHours'] = df['workingHours'].astype(int)
        df['busyHours'] = df['busyHours'].astype(int)

        # identify the sensor faults from the function
        df_only_energy = self.identify_sensor_faults(energy_data)
        df = self.identify_sensor_faults(df)

        # use the above function to reach the cleaned energy data
        df_only_energy = fix_and_compute_energy_difference(df_only_energy, True)
        df = fix_and_compute_energy_difference(df, False)

        # drop the unecessary columns
        df = df.drop(['city', 'time', 'datetime', 'zscore_temp', 'zscore_humi', 'zscore_energy'],
                     axis=1)
        
        df_only_energy = df_only_energy.drop(['zscore_energy'], axis=1)

        # print('After outlier and NaN handling for all data left with: ', df.shape[0], ' rows\n')
        # print('After outlier and NaN handling for energy left with: ', df_only_energy.shape[0], ' rows\n')

        return df, df_only_energy

    # a function that fixes the missing values on temperature and humidity from the indoor sensors
    def handle_missing_on_temp_humi(self):
        # call the preprocessing function to get the data so far
        df, df_only_energy = self.preprocessing()

        # drop the nan values and fill them with forward fill
        list_idx = []
        for idx, value in enumerate(df['temperature']):
            if math.isnan(value):
                list_idx.append(idx)
            else:
                break
        df = df.drop(df.index[list_idx]).reset_index(drop=True)

        # fill nan values with the previous valid value
        df.fillna(method='ffill', inplace=True)

        # make a dataframe for the daily sum of the energy values and add that on the main dataframe
        df2 = df.groupby(['day', 'month', 'year'])[['Energy_Difference']].sum().add_suffix('_daily_sum')
        df = df.join(df2, on=['day', 'month', 'year'])

        # make a histogram to see the frequencies from the whole data 
        attributes = df.iloc[:, :]
        params = {'axes.titlesize': '8'}
        plt.rcParams.update(params)
        ds_attributes = attributes.hist(figsize=(42, 30), xlabelsize=8, ylabelsize=8)  # alpha=0.3, linewidth=2

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'histogramDistribution.png'))

        # Check for seasonality between the outdoor weather attributes and the energy
        data = df[['date', 'maxtempC', 'outdoorHumidity', 'Energy_Difference_daily_sum', 'power']]
        data = data.loc[data.groupby('date')['outdoorHumidity'].idxmax()]
        data.set_index(['date'], inplace=True)

        fig, ax1 = plt.subplots(figsize=(25, 6))
        ax1.plot(data['maxtempC'], color='green', label='Outdoor Temperature')
        ax1.plot(data['outdoorHumidity'], color='blue', label='Outdoor Humidity')
        ax1.plot(data['Energy_Difference_daily_sum'], color='black', label='Daily Energy')
        plt.legend(fontsize='small', fancybox=True, loc='upper left')

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'OutdoorTempHumiEnergySeasonality.png'))

        # Check for seasonality between the indoor weather attributes and the energy
        data = df[['date', 'temperature', 'humidity', 'Energy_Difference_daily_sum', 'power']]
        maxTemp = data.groupby('date')['temperature'].max()
        maxHumi = data.groupby('date')['humidity'].max()
        data['maxtempC'] = data['date'].map(maxTemp)
        data['maxHumi'] = data['date'].map(maxHumi)
        # Update the DataFrame in place
        data.drop(['temperature', 'humidity'], axis=1, inplace=True)
        # dropping duplicate values
        data.drop_duplicates(keep='first', inplace=True)
        data.set_index(['date'], inplace=True)

        fig, ax1 = plt.subplots(figsize=(25, 6))
        ax1.plot(data['maxtempC'], color='green', label='Indoor Temperature')
        ax1.plot(data['maxHumi'], color='blue', label='Indoor Humidity')
        ax1.plot(data['Energy_Difference_daily_sum'], color='black', label='Daily Energy')
        # ax1.plot(data['power'], color='orange', label='Daily Power')
        plt.legend(fontsize='small', fancybox=True, loc='upper left')
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'IndoorTempHumiEnergySeasonality.png'))

        plt.figure()
        plt.xlabel('Observations')
        plt.title('Power over time')
        plt.ylabel('Power Values [W]')
        plt.plot(df["power"])
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'PowerValues.png'))
        
        plt.figure()
        plt.title('Temperature after data cleaning')
        plt.xlabel('Time [hours]')
        plt.ylabel('Temperature [C]')
        plt.plot(df['temperature'])

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'TemperatureDataAfterCleaning.png'))
        
        plt.figure()
        plt.title('Humidity after data cleaning')
        plt.xlabel('Time [hours]')
        plt.ylabel('Humidity [%]')
        plt.plot(df['humidity'])

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'HumidityDataAfterCleaning.png'))

        # check how corelated are the features
        self.correlation(df)

        return df, df_only_energy
    

    # Use principal component analysis to keep only the best features
    def pca(self, X_train, X_test):
        pca = PCA(n_components=10)

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        explained_variance = pca.explained_variance_ratio_
        print('Explained variation per principal component: {}'.format(explained_variance))
        print(f'Explains the : {round(sum(explained_variance), 2)}% of the data')

        return X_train, X_test

    # Function to check the correlation of the features and how depended is one from another
    def correlation(self, final_df):
        corr_threshold = 0.99
        plt.figure(figsize=(14, 12))
        c = final_df.corr()
        mask = np.triu(np.ones_like(c, dtype=bool))
        sns.heatmap(c, mask=mask, annot=False, cmap='coolwarm', linecolor='white', linewidths=0.1)
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'CorrelationHeatMap.png'))
    

    # Function That takes as input the dataframe and the number of days you want to forecast and outputs their corresponding timestamp values
    def find_timestamp(self, df, test_size):
        # get the unique timestamps and short them
        list_of_unique_ts = df["timestamp"].unique()
        last_ts = list_of_unique_ts[np.searchsorted(list_of_unique_ts,list_of_unique_ts, side='right') - 1][-1]

        # number of days for the forecast
        time_window = datetime.timedelta(test_size)
        format_of_ts = "%Y-%m-%dT%H:%M:%S.000000000"
        time_end = datetime.datetime.strptime(str(last_ts), format_of_ts)
        time_end = time_end.replace(hour=00, minute=00)
        time_start = time_end - time_window

        return time_start, time_end
            

    # Function taht splis time series data in train and test set
    def split_data_for_timeseries(self, df, test_size, model, only_energy, scaler=True):
        # For regression and RNN the split is done with hourly data
        if model.lower() == 'regression' or model.lower() == 'rnn':
            # the last number of observations (test size) is the test set and the rest are the train data
            df_train = df[:-test_size]
            df_test = df[-test_size:]

            plt.figure()
            plt.title('Actual & Predicted Window', size=20)
            plt.plot(df_train['Energy_Difference'], label='Training set')
            plt.plot(df_test['Energy_Difference'], label='Test set', color='orange')
            plt.legend()
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory, 'ActualvsPredictedEnergyDifference.png'))

            # Set y the feature of interest
            y_train = df_train['Energy_Difference']
            y_test = df_test['Energy_Difference']
            x_train = df_train.drop(
                ['timestamp', 'energytotal', 'power',
                 'zscore_energyDifference', 
                 'date'],
                axis=1)

            x_test = df_test.drop(
                ['timestamp', 'energytotal', 'power',
                 'zscore_energyDifference',
                 'date'],
                axis=1)

            if only_energy == False:
                x_train = x_train.drop(['Energy_Difference_daily_sum'], axis=1)
                x_test = x_test.drop(['Energy_Difference_daily_sum'], axis=1)
            
            print(x_train.columns)

            # use a scaler to scale the datapoints
            if scaler:
                if self.scaler_f == 'minmax':
                    scaler = MinMaxScaler()
                if self.scaler_f == 'standard':
                    scaler = StandardScaler()

                x_train = scaler.fit_transform(x_train)
                x_test = scaler.fit_transform(x_test)

                # convert to numpy arrays
                y_train = y_train.to_numpy()  # np.array(y_train)
                y_test = y_test.to_numpy()  # np.array(y_test)


            return x_train, x_test, y_train, y_test, df_train, df_test

        # if the model is arima, sarimax or prophet, use the timestamp function to find the timestamps of the train and test set
        # and as data use only the target value
        if model.lower() == 'sarimax' or model.lower() == 'prophet' or model.lower() == 'arima':
            time_start, time_end = self.find_timestamp(df, test_size)

            indexed_energy = df[['timestamp', 'Energy_Difference', 'date', 'year']].set_index('timestamp')

            indexed_energy = indexed_energy.sort_values(by='timestamp', ascending=True)
            indexed_energy.reset_index(inplace=True)

            train = indexed_energy[
                indexed_energy['timestamp'] < time_start]
            test = indexed_energy[indexed_energy['timestamp'] >= time_start]

            train.set_index('timestamp', inplace=True)
            test.set_index('timestamp', inplace=True)

            return train, test, indexed_energy
        
        # if the model is sarimax with exogeneous data, use the timestamp function to find the timestamps of the train and test set
        # and as data use also some other data from whole dataset
        if model.lower() == 'sarimax-exog':
            time_start, time_end = self.find_timestamp(df, test_size)
            print(df.columns)
            indexed_energy = df[['timestamp', 'Energy_Difference', 'date', 'year']].set_index('timestamp')
            exogeneous_data = df[['timestamp', 'temperature', 'humidity', 'date', 'year']].set_index('timestamp')

            indexed_energy = indexed_energy.sort_values(by='timestamp', ascending=True)
            indexed_energy.reset_index(inplace=True)
            exogeneous_data = exogeneous_data.sort_values(by='timestamp', ascending=True)
            exogeneous_data.reset_index(inplace=True)

            train = indexed_energy[
                indexed_energy['timestamp'] < time_start]  # train set is years 2020 - 2021
            test = indexed_energy[indexed_energy['timestamp'] >= time_start]  # test set is year 2022
            exogeneous_train = exogeneous_data[
                exogeneous_data['timestamp'] < time_start]  # train set is years 2020 - 2021
            exogeneous_test = exogeneous_data[exogeneous_data['timestamp'] >= time_start]  # test set is year 2022

            train.set_index('timestamp', inplace=True)
            test.set_index('timestamp', inplace=True)
            exogeneous_train.set_index('timestamp', inplace=True)
            exogeneous_test.set_index('timestamp', inplace=True)

            return train, test, indexed_energy, exogeneous_train, exogeneous_test


    def train_Regression(self, test_size, only_energy):
        # get the dataframes (one taht has only energy values and one with all the parameters)
        df, df_only_energy = self.handle_missing_on_temp_humi()
        if only_energy==True:
            df = df_only_energy
        
        # test size is test_size (days) * r4 hours per day
        test_size = test_size * 24

        # split the data
        x_train, x_test, y_train, y_test, df_train, df_test = self.split_data_for_timeseries(df, test_size, 'regression', only_energy)

        # x_train, x_test = self.pca(x_train, x_test)

        # initiate linear regression fit the data and make prediction
        lm = LinearRegression()
        model = lm.fit(x_train, y_train)
        predictions = lm.predict(x_test)

        # get predictions for train and test
        y_train_pred = lm.predict(x_train)
        y_train_pred_rnn = np.reshape(y_train_pred, (y_train.shape[0]))
        y_test_pred = lm.predict(x_test)
        y_test_pred_rnn = np.reshape(y_test_pred, (y_test.shape[0]))

        df_train['predictions'] = y_train_pred_rnn
        df_train = df_train.reset_index()
     
        df_test['predictions'] = y_test_pred_rnn
        df_test = df_test.reset_index()

        # sum the values daily to output the daily prediction the actual values and their residuals for test and train
        ts_24 = df_test['timestamp'].groupby(df_test['timestamp'].index // 24).last()
        actual_24 = df_test['Energy_Difference'].groupby(df_test['Energy_Difference'].index // 24).sum()
        prediction_24 = df_test['predictions'].groupby(df_test['predictions'].index // 24).sum()
        df_test = pd.merge(actual_24, prediction_24, left_index=True, right_index=True)
        df_test = pd.merge(ts_24, df_test, left_index=True, right_index=True)
        df_test.insert(3, 'residuals', df_test['Energy_Difference'] - df_test['predictions'])

        train_ts_24 = df_train['timestamp'].groupby(df_train['timestamp'].index // 24).last()
        train_actual_24 = df_train['Energy_Difference'].groupby(df_train['Energy_Difference'].index // 24).sum()
        train_prediction_24 = df_train['predictions'].groupby(df_train['predictions'].index // 24).sum()
        df_train = pd.merge(train_actual_24, train_prediction_24, left_index=True, right_index=True)
        df_train = pd.merge(train_ts_24, df_train, left_index=True, right_index=True)
        df_train.insert(3, 'residuals', df_train['Energy_Difference'] - df_train['predictions'])

        # merge the two dataframes
        forecast = pd.merge(df_train, df_test, how='outer')

        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(15, 7)
        plt.plot(df_test.timestamp, df_test.Energy_Difference, '.', color='green', label = "test_actual")
        plt.plot(df_train.timestamp, df_train.Energy_Difference, '.', color='#3498db', label = "train_actual")
        plt.plot(forecast.timestamp, forecast.predictions, color='black', label = "forecast")
        plt.grid(color=(0, 0, 0), linestyle='-', linewidth=1, alpha=0.05)
        plt.xlabel('Energy_Difference')
        plt.title('Daily Energy: Actual Values vs Regression Prediction')
        plt.ylabel('Date (Daily)')
        plt.legend()

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'RegressionPredictionGraph.png'))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, predictions)

        plt.figure()
        plt.plot(y_test, predictions, 'o', label='data')
        plt.plot(y_test, intercept + slope * y_test, 'r', label='fitted line')
        plt.legend()
        plt.annotate(
            'y = ' + str(round(slope, 2)) + 'x' + ' + ' + str(round(intercept, 2)), xy=(3, 1.5), xytext=(3.2, 1.25),
            arrowprops=dict(facecolor='black', shrink=0.01))
        plt.xlabel('Actual Values')
        plt.title('Actual vs Predicted Values')
        plt.ylabel('Predicted Values')

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'Regression_ActualvsPredictedValuesFitLine.png'))

        # MSE
        mse = np.mean(df_test['residuals'] ** 2)
        print("TEST MSE = ", mse)
        # RMSE
        rmse = round(np.sqrt(mse), 2)
        print('TEST RMSE = ', rmse)
        # MAPE
        mape = np.mean(np.abs(df_test['residuals'])/df_test['Energy_Difference'])*100
        print('TEST MAPE = ', mape, '%')
        # R2
        r_squared = r2_score(df_test['Energy_Difference'].tolist(), df_test['predictions'].tolist())
        print("TEST R2 = ", r_squared)

        # MSE
        train_mse = np.mean(df_train['residuals'] ** 2)
        print("TRAIN MSE = ", train_mse)
        # RMSE
        train_rmse = round(np.sqrt(train_mse), 2)
        print('TRAIN RMSE = ', train_rmse)
        # MAPE
        train_mape = np.mean(np.abs(df_train['residuals'])/df_train['Energy_Difference'])*100
        print('TRAIN MAPE = ', train_mape, '%')
        # R2
        train_r_squared = r2_score(df_train['Energy_Difference'].tolist(), df_train['predictions'].tolist())
        print("TRAIN R2 = ", r_squared)

        end = int(time.time() - self.start_time)
        print(df_test)
        print(f'Model finished in {end} seconds')

        # Print the results
        if self.save_evaluation == True:
            res = {"mse":[mse], "rmse":[rmse],"mape":[mape], "r2":[r_squared], "time_in_seconds": [end]}
            res = pd.DataFrame(res)
            res.to_csv(os.path.join(self.evaluation_directory, 'Regression_Test_Evaluation.csv'), index=False)
            res_train = {"mse":[train_mse], "rmse":[train_rmse],"mape":[train_mape], "r2":[train_r_squared], "time_in_seconds": [end]}
            res_train = pd.DataFrame(res_train)
            res_train.to_csv(os.path.join(self.evaluation_directory, 'Regression_Train_Evaluation.csv'), index=False)

        if self.save_predictions == True:
            df_test.to_csv(os.path.join(self.predictions_directory, 'Regression_Test_Predictions.csv'), index=False)
            df_train.to_csv(os.path.join(self.predictions_directory, 'Regression_Train_Predictions.csv'), index=False)

        if self.show_plots:
            plt.show()
    

    def train_SarimaX(self, test_size, only_energy, exogeneous_data, grid_search_for_order_params):
        # get the dataframes (one taht has only energy values and one with all the parameters)
        df, df_only_energy = self.handle_missing_on_temp_humi()
        
        # take the correct dataframe (only energy data or the whole dataset)
        if only_energy==True:
            df_main = df_only_energy
        else:
            df_main = df
        
        # find the timestamps for the train and test
        time_start, time_end = self.find_timestamp(df_main, test_size)

        # if exogeneous flag is true split data acording to sarimax with external data else normally
        if exogeneous_data==True and only_energy==False:
            train, test, indexed_energy, exogeneous_train, exogeneous_test = self.split_data_for_timeseries(df_main, test_size, 'sarimax-exog', only_energy)
        else:
            train, test, indexed_energy = self.split_data_for_timeseries(df_main, test_size, 'sarimax', only_energy)

        # group the data daily (compute consumption per day)
        train_daily = train[['Energy_Difference']].resample('D').sum()  # daily prediction  

        # if external flag is on group also temp and humidity, taking the mean of the day
        if exogeneous_data==True and only_energy==False: 
            exogeneous_train = exogeneous_train[['temperature', 'humidity']].resample('D').mean()
            exogeneous_test = exogeneous_test[['temperature', 'humidity']].resample('D').mean()

        indexed_energy = indexed_energy.set_index('timestamp')
        indexed_energy = indexed_energy[['Energy_Difference']].resample('D').sum().reset_index()  # daily prediction

        # function that checks if the date are stationary or not
        def check_stationarity(ts):
            dftest = adfuller(ts)
            adf = dftest[0]
            pvalue = dftest[1]
            critical_value = dftest[4]['5%']
            if (pvalue < 0.05) and (adf < critical_value):
                print('The series is stationary')
            else:
                print('The series is NOT stationary')

        # Deciding (P,D,Q,M) Order:
        # function that trains the model with all the combinations of pdq and keeps the one with the least AIC
        def choosing_params(train_daily, test_size):
            p  = q = range(0,3)
            d = range(0,2)
            pdq = list(itertools.product(p, d, q))
            seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
            count = 0
            list_of_params = []
            for param in pdq:
                for param_seasonal in seasonal_pdq:
                    count+=1
                    print(count, 'out of ', len(pdq)*len(seasonal_pdq))
                    try:
                        mod = sm.tsa.statespace.SARIMAX(train_daily, order=param, seasonal_order=param_seasonal,
                                                        enforce_stationarity=False, enforce_invertibility=False)
                        results = mod.fit()
                        # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                        list_of_params.append([param, param_seasonal, results.aic])
                    except:
                        continue
            
            list_of_params.sort(key=lambda x: x[2], reverse=False)
            best_order = tuple(list_of_params[0][0])
            best_seasonal_order = tuple(list_of_params[0][1])
            # print(best_order, best_seasonal_order)

            return best_order, best_seasonal_order
            
        # Manual setting of model parameters and multi-step forecasting
        def pdqm_order(indexed_energy, test_size):
            indexed_energy.set_index('timestamp', inplace=True)
            # print(indexed_energy.head())
            indexed_energy = indexed_energy[['Energy_Difference']]
            result = seasonal_decompose(indexed_energy, model='additive', extrapolate_trend='freq')
            result.plot()
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory ,'SarimaSeasonalTrend.png'))

            # D order
            seasonal = result.seasonal
            check_stationarity(seasonal)
            # Since the series is stationary, we do not need any additional transformation to make it stationary.
            # We can set D = 0.

            # P order
            plot_pacf(seasonal, lags=24)  # lags=92
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory, 'SarimaPACFvsLag.png'))

            # Q order
            plot_acf(seasonal, lags=24)  # lags=92
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory, 'SarimaACFvsLag.png'))

        # if grid search flag is one use the above function, else the best parameters discovered through trial and error will be kept
        if grid_search_for_order_params == True:
            best_order, best_seasonal_order = choosing_params(train_daily, test_size)
        else:
            best_order, best_seasonal_order = (0, 1, 0), (1, 1, 1, 7)

        pdqm_order(indexed_energy, test_size)

        # fit the model with the hyperparameters and make the predictions
        if exogeneous_data==True and only_energy==False:
            model = SARIMAX(train_daily,
                            order=best_order,
                            seasonal_order=best_seasonal_order,
                            time_varying_regression = False,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            exog=exogeneous_train)
            results = model.fit(disp=0)

            results.plot_diagnostics(figsize=(18, 8))
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory, 'SarimaDiagnostics.png'))

            time_from_beggining = train_daily.index[0]
            predictions = results.predict(start=time_start, end=time_end, exog=exogeneous_test, dynamic=True)
            predictions_train = results.predict(start=time_from_beggining, end=time_end, exog=exogeneous_test, dynamic=True)
        else:
            model = SARIMAX(train_daily,
                            order=best_order,
                            seasonal_order=best_seasonal_order,
                            trend = 'n',
                            measurement_error = False,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=0)

            results.plot_diagnostics(figsize=(18, 8))
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory, 'SarimaDiagnostics.png'))

            time_from_beggining = train_daily.index[0]
            predictions = results.predict(start=time_start, end=time_end)
            predictions_train = results.predict(start=time_from_beggining, end=time_end)

        # merge the data in dataframes and compute the residuals 
        # will be used for graphs
        predictions_train = pd.DataFrame(predictions_train).reset_index()
        predictions_train.columns = ['timestamp', 'Energy_Difference']

        # merge on timestamp
        test_daily = test[['Energy_Difference']].resample('D').sum().reset_index()

        prediction = pd.DataFrame(predictions).reset_index()
        prediction.columns = ['timestamp', 'Energy_Difference']

        res = pd.merge(test_daily, prediction, how='left', on='timestamp')

        res.columns = ['timestamp', 'actual', 'predictions']
        res.insert(3, 'residuals', res['actual'] - res['predictions'])  # residuals
        res.head()

        res_train = pd.merge(train_daily, predictions_train, how='left', on='timestamp')

        res_train.columns = ['timestamp', 'actual', 'predictions']
        res_train.insert(3, 'residuals', res_train['actual'] - res_train['predictions'])  # residuals
        res_train.head()

        f, axes = plt.subplots(2, figsize=(15, 10), sharex=True)

        # plot of actual vs predictions
        axes[0].plot(train_daily.index, train_daily['Energy_Difference'], color='blue', label='train actual')
        axes[0].plot(res['timestamp'], res['actual'], color='green', label='test actual')
        axes[0].plot(res['timestamp'], res['predictions'], color='black', label='prediction')
        axes[0].set_title('Actual vs Predicted Energy')
        axes[0].set_ylabel('Energy')
        axes[0].legend()

        # plot of actual - predictions
        axes[1].scatter(res['timestamp'], (res['actual'] - res['predictions']))
        axes[1].set_title('Residual Plot')
        axes[1].set_xlabel('Date (By Week)')
        axes[1].set_ylabel('Actuals - Predictions')
        axes[1].axhline(y=0, color='r', linestyle=':')
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'SarimaActualValuesAndTestPrediction.png'))
        

        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(15, 7)
        plt.plot(test_daily.timestamp, test_daily.Energy_Difference, '.', color='green', label = "test_actual")
        plt.plot(train_daily.index, train_daily.Energy_Difference, '.', color='#3498db', label = "train_actual")
        plt.plot(predictions_train.timestamp, predictions_train.Energy_Difference, color='black', label = "forecast")
        plt.grid(color=(0, 0, 0), linestyle='-', linewidth=1, alpha=0.05)
        plt.xlabel('Energy_Difference')
        plt.title('Daily Energy: Actual vs Sarima Prediction')
        plt.ylabel('Date (Daily)')
        plt.legend()

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'SarimaTestTrainPrediction.png'))
        
        # MSE
        mse = np.mean(res['residuals'] ** 2)
        print("TEST MSE = ", mse)
        # RMSE
        rmse = round(np.sqrt(mse), 2)
        print('TEST RMSE = ', rmse)
        # MAPE
        mape = np.mean(np.abs(res['residuals'])/res['actual'])*100
        print('TEST MAPE = ', mape, '%')
        # r2
        r_squared = r2_score(res['actual'].tolist(), res['predictions'].tolist())
        print('TEST R2 = ', r_squared)

        # MSE
        train_mse = np.mean(res_train['residuals'] ** 2)
        print("TRAIN MSE = ", train_mse)
        # RMSE
        train_rmse = round(np.sqrt(train_mse), 2)
        print('TRAIN RMSE = ', train_rmse)
        # MAPE
        train_mape = np.mean(np.abs(res_train['residuals'])/res_train['actual'])*100
        print('TRAIN MAPE = ', train_mape, '%')
        # r2
        train_r_squared = r2_score(res_train['actual'].tolist(), res_train['predictions'].tolist())
        print('TRAIN R2 = ', train_r_squared)

        # Print the results
        print(res)
        end = int(time.time() - self.start_time)
        print(f'Model finished in {end} seconds')

        # Print the results
        if self.save_evaluation == True:
            df = {"mse":[mse], "rmse":[rmse],"mape":[mape], "r2":[r_squared], "time_in_seconds": [end]}
            df = pd.DataFrame(df)
            df.to_csv(os.path.join(self.evaluation_directory, 'SarimaX_Test_Evaluation.csv'), index=False)
            df_train = {"mse":[train_mse], "rmse":[train_rmse],"mape":[train_mape], "r2":[train_r_squared], "time_in_seconds": [end]}
            df_train = pd.DataFrame(df_train)
            df_train.to_csv(os.path.join(self.evaluation_directory, 'SarimaX_Train_Evaluation.csv'), index=False)

        if self.save_predictions == True:
            res.to_csv(os.path.join(self.predictions_directory, 'SarimaX_Test_Predictions.csv'), index=False)
            res_train.to_csv(os.path.join(self.predictions_directory, 'SarimaX_Train_Predictions.csv'), index=False)

        if self.show_plots:
            plt.show()
    

    def train_Arima(self, test_size, only_energy):
        # get the dataframes (one taht has only energy values and one with all the parameters)
        df, df_only_energy = self.handle_missing_on_temp_humi()
        
        # take the correct dataframe (only energy data or the whole dataset)
        if only_energy==True:
            df_main = df_only_energy
        else:
            df_main = df

        # find the timestamps for the train and test
        time_start, time_end = self.find_timestamp(df_main, test_size)

        # if exogeneous flag is true split data acording to sarimax with external data else normally
        train, test, indexed_energy = self.split_data_for_timeseries(df_main, test_size, 'arima', only_energy)

        # group the data daily (compute consumption per day)
        train_daily = train[['Energy_Difference']].resample('D').sum()  # daily prediction  

        indexed_energy = indexed_energy.set_index('timestamp')
        indexed_energy = indexed_energy[['Energy_Difference']].resample('D').sum().reset_index()  # daily prediction

        def check_stationarity(ts):
            dftest = adfuller(ts)
            adf = dftest[0]
            pvalue = dftest[1]
            critical_value = dftest[4]['5%']
            if (pvalue < 0.05) and (adf < critical_value):
                print('The series is stationary')
            else:
                print('The series is NOT stationary')

        # Manual setting of model parameters and multi-step forecasting
        def pdqm_order(indexed_energy, test_size):
            indexed_energy.set_index('timestamp', inplace=True)
            indexed_energy = indexed_energy[['Energy_Difference']]
            result = seasonal_decompose(indexed_energy, model='additive', extrapolate_trend='freq')
            result.plot()
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory ,'ArimaSeasonalTrend.png'))

            # D order
            seasonal = result.seasonal
            check_stationarity(seasonal)
            # Since the series is stationary, we do not need any additional transformation to make it stationary.
            # We can set D = 0.

            # P order
            plot_pacf(seasonal, lags=24)  # lags=92
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory, 'ArimaPACFvsLag.png'))

            # Q order
            plot_acf(seasonal, lags=24)  # lags=92
            if self.save_plots == True:
                plt.savefig(os.path.join(self.plot_directory, 'ArimaACFvsLag.png'))

        pdqm_order(indexed_energy, test_size)

        # fit the model with the hyperparameters and make the predictions
        model = ARIMA(train_daily,
                        order=(0, 1, 0))
        results = model.fit()

        results.plot_diagnostics(figsize=(18, 8))
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'ArimaDiagnostics.png'))

        time_from_beggining = train_daily.index[0]
        predictions = results.predict(start=time_start, end=time_end)
        predictions_train = results.predict(start=time_from_beggining, end=time_end)

        predictions_train = pd.DataFrame(predictions_train).reset_index()
        predictions_train.columns = ['timestamp', 'Energy_Difference']

        # merge on timestamp
        test_daily = test[['Energy_Difference']].resample('D').sum().reset_index()

        prediction = pd.DataFrame(predictions).reset_index()
        prediction.columns = ['timestamp', 'Energy_Difference']

        res = pd.merge(test_daily, prediction, how='left', on='timestamp')

        res.columns = ['timestamp', 'actual', 'predictions']
        res.insert(3, 'residuals', res['actual'] - res['predictions'])  # residuals
        res.head()

        res_train = pd.merge(train_daily, predictions_train, how='left', on='timestamp')

        res_train.columns = ['timestamp', 'actual', 'predictions']
        res_train.insert(3, 'residuals', res_train['actual'] - res_train['predictions'])  # residuals
        res_train.head()

        f, axes = plt.subplots(2, figsize=(15, 10), sharex=True)

        # plot of actual vs predictions
        axes[0].plot(train_daily.index, train_daily['Energy_Difference'], color='blue', label='train actual')
        axes[0].plot(res['timestamp'], res['actual'], color='green', label='test actual')
        axes[0].plot(res['timestamp'], res['predictions'], color='black', label='prediction')
        axes[0].set_title('Actual vs Predicted Energy')
        axes[0].set_ylabel('Energy')
        axes[0].legend()

        # plot of actual - predictions
        axes[1].scatter(res['timestamp'], (res['actual'] - res['predictions']))
        axes[1].set_title('Residual Plot')
        axes[1].set_xlabel('Date (By Week)')
        axes[1].set_ylabel('Actuals - Predictions')
        axes[1].axhline(y=0, color='r', linestyle=':')
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'ArimaActualValuesAndTestPrediction.png'))
        

        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(15, 7)
        plt.plot(test_daily.timestamp, test_daily.Energy_Difference, '.', color='green', label = "test_actual")
        plt.plot(train_daily.index, train_daily.Energy_Difference, '.', color='#3498db', label = "train_actual")
        plt.plot(predictions_train.timestamp, predictions_train.Energy_Difference, color='black', label = "forecast")
        # plt.fill_between(forecast.ds, forecast.yhat_lower, forecast.yhat_upper, color=(52/255, 152/255, 219/255, 0.2))
        plt.grid(color=(0, 0, 0), linestyle='-', linewidth=1, alpha=0.05)
        plt.xlabel('Energy_Difference')
        plt.title('Daily Energy: Actual vs Arima Prediction')
        plt.ylabel('Date (Daily)')
        plt.legend()

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'ArimaTestTrainPrediction.png'))
        

        # MSE
        mse = np.mean(res['residuals'] ** 2)
        print("TEST MSE = ", mse)
        # RMSE
        rmse = round(np.sqrt(mse), 2)
        print('TEST RMSE = ', rmse)
        # MAPE
        mape = np.mean(np.abs(res['residuals'])/res['actual'])*100
        print('TEST MAPE = ', mape, '%')
        # r2
        r_squared = r2_score(res['actual'].tolist(), res['predictions'].tolist())
        print('TEST R2 = ', r_squared)

        # MSE
        train_mse = np.mean(res_train['residuals'] ** 2)
        print("TRAIN MSE = ", train_mse)
        # RMSE
        train_rmse = round(np.sqrt(train_mse), 2)
        print('TRAIN RMSE = ', train_rmse)
        # MAPE
        train_mape = np.mean(np.abs(res_train['residuals'])/res_train['actual'])*100
        print('TRAIN MAPE = ', train_mape, '%')
        # r2
        train_r_squared = r2_score(res_train['actual'].tolist(), res_train['predictions'].tolist())
        print('TRAIN R2 = ', train_r_squared)

        # Print the results
        print(res)
        end = int(time.time() - self.start_time)
        print(f'Model finished in {end} seconds')

        # Save the results
        if self.save_evaluation == True:
            df = {"mse":[mse], "rmse":[rmse],"mape":[mape], "r2":[r_squared], "time_in_seconds": [end]}
            df = pd.DataFrame(df)
            df.to_csv(os.path.join(self.evaluation_directory, 'Arima_Test_Evaluation.csv'), index=False)
            df_train = {"mse":[train_mse], "rmse":[train_rmse],"mape":[train_mape], "r2":[train_r_squared], "time_in_seconds": [end]}
            df_train = pd.DataFrame(df_train)
            df_train.to_csv(os.path.join(self.evaluation_directory, 'Arima_Train_Evaluation.csv'), index=False)

        if self.save_predictions == True:
            res.to_csv(os.path.join(self.predictions_directory, 'Arima_Test_Predictions.csv'), index=False)
            res_train.to_csv(os.path.join(self.predictions_directory, 'Arima_Train_Predictions.csv'), index=False)

        if self.show_plots:
            plt.show()


    def train_Rnn(self, test_size, only_energy, epochs, batch_size, optimizer):
        def baseline_model():
            # design network with lstm layers, dropout layers and dense layers
            model = tf.keras.models.Sequential()
            model.add(
                tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]),
                                     dropout=0.01))
            model.add(
                tf.keras.layers.LSTM(units=80, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]),
                                     dropout=0.02))
            model.add(tf.keras.layers.Dense(units=128, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(tf.keras.layers.Dense(units=64, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(Dense(16, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(Dense(1, activation='relu'))

            print(model.summary())
            # es = EarlyStopping(monitor='mae', mode=100, verbose=4)
            # es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, verbose=4)  # , patience=3)
            return model

        # take the dataset
        df, df_only_energy = self.handle_missing_on_temp_humi()
        if only_energy==True:
            df = df_only_energy

        # test size is days* 24 hours per day
        test_size = test_size * 24

        # split the data
        x_train, x_test, y_train, y_test, df_train, df_test = self.split_data_for_timeseries(df, test_size, 'rnn', only_energy)
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        # construct the hyperparameters
        model = baseline_model()
        if optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop()
        if optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam()
        if optimizer == 'adadelta':
            opt = tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta")
        if optimizer == 'adagrad':
            opt = tf.keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
        if optimizer == 'adamax':
            opt = tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")
        
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])

        # fit the model and make the predictions
        history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_test, y_test), shuffle=False)  # , callbacks=[es])
        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        y_train_pred = model.predict(x_train)
        y_train_pred_rnn = np.reshape(y_train_pred, (y_train.shape[0]))

        df_train['predictions'] = y_train_pred_rnn
        df_train = df_train.reset_index()
        # print(df_train[['Energy_Difference', 'predictions']])
 
        fig, ax1 = plt.subplots(figsize=(40, 15))
        ax1.plot(y_train_pred_rnn, color='blue')
        ax1.plot(y_train, color='red')
        ax1.legend(('y_train_pred_rnn', 'y_train'))
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'RNNTrainPrediction.png'))

        y_test_pred = model.predict(x_test)
        y_test_pred_rnn = np.reshape(y_test_pred, (y_test.shape[0]))

        df_test['predictions'] = y_test_pred_rnn
        df_test = df_test.reset_index()
        # print(df_test[['Energy_Difference', 'predictions']])

        # sum the values daily to output the daily prediction the actual values and their residuals for test and train
        ts_24 = df_test['timestamp'].groupby(df_test['timestamp'].index // 24).last()
        actual_24 = df_test['Energy_Difference'].groupby(df_test['Energy_Difference'].index // 24).sum()
        prediction_24 = df_test['predictions'].groupby(df_test['predictions'].index // 24).sum()
        df_test = pd.merge(actual_24, prediction_24, left_index=True, right_index=True)
        df_test = pd.merge(ts_24, df_test, left_index=True, right_index=True)
        df_test.insert(3, 'residuals', df_test['Energy_Difference'] - df_test['predictions'])

        train_ts_24 = df_train['timestamp'].groupby(df_train['timestamp'].index // 24).last()
        train_actual_24 = df_train['Energy_Difference'].groupby(df_train['Energy_Difference'].index // 24).sum()
        train_prediction_24 = df_train['predictions'].groupby(df_train['predictions'].index // 24).sum()
        df_train = pd.merge(train_actual_24, train_prediction_24, left_index=True, right_index=True)
        df_train = pd.merge(train_ts_24, df_train, left_index=True, right_index=True)
        df_train.insert(3, 'residuals', df_train['Energy_Difference'] - df_train['predictions'])

        forecast = pd.merge(df_train, df_test, how='outer')

        # MSE
        mse = np.mean(df_test['residuals'] ** 2)
        print("TEST MSE = ", mse)
        # RMSE
        rmse = round(np.sqrt(mse), 2)
        print('TEST RMSE = ', rmse)
        # MAPE
        mape = np.mean(np.abs(df_test['residuals'])/df_test['Energy_Difference'])*100
        print('TEST MAPE = ', mape, '%')
        # r2
        r_squared = r2_score(df_test['Energy_Difference'].tolist(), df_test['predictions'].tolist())
        print('TEST R2 = ', r_squared)

        # MSE
        train_mse = np.mean(df_train['residuals'] ** 2)
        print("TRAIN MSE = ", train_mse)
        # RMSE
        train_rmse = round(np.sqrt(train_mse), 2)
        print('TRAIN RMSE = ', train_rmse)
        # MAPE
        train_mape = np.mean(np.abs(df_train['residuals'])/df_train['Energy_Difference'])*100
        print('TRAIN MAPE = ', train_mape, '%')
        # r2
        train_r_squared = r2_score(df_train['Energy_Difference'].tolist(), df_train['predictions'].tolist())
        print('TRAIN R2 = ', train_r_squared)

        end = int(time.time() - self.start_time)
        print(f'Model finished in {end} seconds')

        # Print the results
        if self.save_evaluation == True:
            res = {"mse":[mse], "rmse":[rmse],"mape":[mape], "r2":[r_squared], "time_in_seconds": [end]}
            res = pd.DataFrame(res)
            res.to_csv(os.path.join(self.evaluation_directory, 'RNN_Test_Evaluation.csv'), index=False)
            res_train = {"mse":[train_mse], "rmse":[train_rmse],"mape":[train_mape], "r2":[train_r_squared], "time_in_seconds": [end]}
            res_train = pd.DataFrame(res_train)
            res_train.to_csv(os.path.join(self.evaluation_directory, 'RNN_Train_Evaluation.csv'), index=False)

        # compute train and test loss vs epochs
        epochs = range(1, epochs+1)
        plt.figure()
        plt.plot(epochs, loss_train, 'g', label='Training loss')
        plt.plot(epochs, loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'RNNLossvsEpochs.png'))
        
        f, axes = plt.subplots(2, figsize=(15, 10), sharex=True)

        # plot of actual vs predictions
        axes[0].plot(df_test['timestamp'], df_test['Energy_Difference'], color='black', label='actual')
        axes[0].plot(df_test['timestamp'], df_test['predictions'], color='blue', label='prediction')
        axes[0].set_title('Actual vs Predicted Energy')
        axes[0].set_ylabel('Energy')
        axes[0].legend()

        # plot of actual - predictions
        axes[1].scatter(df_test['timestamp'],df_test['Energy_Difference'] - df_test['predictions'])
        axes[1].set_title('Residual Plot')
        axes[1].set_xlabel('Date (By Week)')
        axes[1].set_ylabel('Actuals - Predictions')
        axes[1].axhline(y=0, color='r', linestyle=':')
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'RNNPrediction&residuals.png'))
        
        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(15, 7)
        plt.plot(df_test.timestamp, df_test.Energy_Difference, '.', color='green', label = "test_actual")
        plt.plot(df_train.timestamp, df_train.Energy_Difference, '.', color='#3498db', label = "train_actual")
        plt.plot(forecast.timestamp, forecast.predictions, color='black', label = "forecast")
        # plt.fill_between(forecast.ds, forecast.yhat_lower, forecast.yhat_upper, color=(52/255, 152/255, 219/255, 0.2))
        plt.grid(color=(0, 0, 0), linestyle='-', linewidth=1, alpha=0.05)
        plt.xlabel('Energy_Difference')
        plt.title('Daily Energy: Actual vs RNN Prediction')
        plt.ylabel('Date (Daily)')
        plt.legend()
        
        if self.save_predictions == True:
            df_test.to_csv(os.path.join(self.predictions_directory, 'RNN_Test_Predictions.csv'), index=False)
            df_train.to_csv(os.path.join(self.predictions_directory, 'RNN_Train_Predictions.csv'), index=False)

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'RNNPrediction.png'))

        if self.show_plots:
            plt.show()

    
    def train_Prophet(self, test_size, only_energy, add_holidays, period, fourier_order):
        # get the dataframes (one taht has only energy values and one with all the parameters)
        df, df_only_energy = self.handle_missing_on_temp_humi()
        
        # take the correct dataframe (only energy data or the whole dataset)
        if only_energy==True:
            df_main = df_only_energy
        else:
            df_main = df
        
        # module takes also into consideration the bank holidays in Greece
        gr_holidays = holidays.Greece()
        holiday_df = df[['timestamp']]
        holiday_df['holiday'] = pd.Series(holiday_df['timestamp']).apply(lambda x: gr_holidays.get(x)).values
        holiday_df = holiday_df.replace(to_replace='None', value=np.nan).dropna()
        holiday_df['ds'] = pd.to_datetime(holiday_df['timestamp'])

        # find the timestamps for the train and test
        time_start, time_end = self.find_timestamp(df_main, test_size)

        # split the data in train and test based on the timestamps
        train, test, indexed_energy = self.split_data_for_timeseries(df_main, test_size, 'prophet', only_energy)

        # group the data daily (compute consumption per day)
        train_daily = train[['Energy_Difference']].resample('D').sum()  # daily prediction   

        indexed_energy = indexed_energy.set_index('timestamp')
        indexed_energy = indexed_energy[['Energy_Difference']].resample('D').sum().reset_index()  # daily prediction

        train_daily.reset_index(inplace=True)
        train_daily = train_daily[['timestamp', 'Energy_Difference']]
        train_daily.columns = ['ds', 'y']
        
        test_daily = test.resample('D').sum()
        test_daily.reset_index(inplace=True)
        test_daily = test_daily[['timestamp', 'Energy_Difference']]
        test_daily.columns = ['ds', 'y']
        
        # configure the model
        if add_holidays == True:
            ph = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=True, holidays=holiday_df, holidays_prior_scale = 0.05)
            ph.add_country_holidays(country_name='GR')
        else:
            ph = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=True)

        # add seasonality term and fit-predict
        ph.add_seasonality(name='daily', period=period, fourier_order=fourier_order) # original fourier_order=8
        ph.fit(train_daily)
        
        future = ph.make_future_dataframe(periods=test_size+1, include_history=True)
        forecast = ph.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
        
        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(15, 7)
        plt.plot(test_daily.ds, test_daily.y, '.', color='green', label = "test_actual")
        plt.plot(train_daily.ds, train_daily.y, '.', color='#3498db', label = "train_actual")
        plt.plot(forecast.ds, forecast.yhat, color='black', label = "forecast")
        plt.fill_between(forecast.ds, forecast.yhat_lower, forecast.yhat_upper, color=(52/255, 152/255, 219/255, 0.2))
        plt.grid(color=(0, 0, 0), linestyle='-', linewidth=1, alpha=0.05)
        plt.xlabel('Energy_Difference')
        plt.title('Daily Energy: Actual vs Prophet Prediction')
        plt.ylabel('Date (Daily)')
        plt.legend()

        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'ProphetPrediction.png'))

        # merge on Date_Time
        res = pd.merge(test_daily, 
                    forecast[['ds','yhat']], 
                    how='left', 
                    on='ds')
        res.columns = ['timestamp','actual','predictions']
        res.insert(3, 'residuals', res['actual'] - res['predictions']) #residuals
        ## print(res2.head())

        # merge on Date_Time
        res_train = pd.merge(train_daily, 
                    forecast[['ds','yhat']], 
                    how='left', 
                    on='ds')
        res_train.columns = ['timestamp','actual','predictions']
        res_train.insert(3, 'residuals', res_train['actual'] - res_train['predictions']) #residuals
        ## print(res2.head())
        
        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(15, 7)
        
        #plot of residuals
        plt.scatter(res['timestamp'],(res['residuals']))
        plt.title('Residual Plot')
        plt.xlabel('Date (By Week)')
        plt.ylabel('Actuals - Predictions')
        plt.axhline(y=0, color='r', linestyle=':')
        
        if self.save_plots == True:
            plt.savefig(os.path.join(self.plot_directory, 'ProphetResiduals.png'))

        # MSE
        mse = np.mean(res['residuals'] ** 2)
        print("TEST MSE = ", mse)
        # RMSE
        rmse = round(np.sqrt(mse), 2)
        print('TEST RMSE = ', rmse)
        # MAPE
        mape = np.mean(np.abs(res['residuals'])/res['actual'])*100
        print('TEST MAPE = ', mape, '%')
        # r2
        r_squared = r2_score(res['actual'].tolist(), res['predictions'].tolist())
        print('TEST R2 = ', r_squared)

        # MSE
        train_mse = np.mean(res_train['residuals'] ** 2)
        print("TRAIN MSE = ", train_mse)
        # RMSE
        train_rmse = round(np.sqrt(train_mse), 2)
        print('TRAIN RMSE = ', train_rmse)
        # MAPE
        train_mape = np.mean(np.abs(res_train['residuals'])/res_train['actual'])*100
        print('TRAIN MAPE = ', train_mape, '%')
        # r2
        train_r_squared = r2_score(res_train['actual'].tolist(), res_train['predictions'].tolist())
        print('TRAIN R2 = ', train_r_squared)

        # Print the results
        print(res)
        end = int(time.time() - self.start_time)
        print(f'Model finished in {end} seconds')

        # Print the results
        if self.save_evaluation == True:
            df = {"mse":[mse], "rmse":[rmse],"mape":[mape], "r2":[r_squared], "time_in_seconds": [end]}
            df = pd.DataFrame(df)
            df.to_csv(os.path.join(self.evaluation_directory, 'Prophet_Test_Evaluation.csv'), index=False)
            df_train = {"mse":[train_mse], "rmse":[train_rmse],"mape":[train_mape], "r2":[train_r_squared], "time_in_seconds": [end]}
            df_train = pd.DataFrame(df_train)
            df_train.to_csv(os.path.join(self.evaluation_directory, 'Prophet_Train_Evaluation.csv'), index=False)

        if self.save_predictions == True:
            res.to_csv(os.path.join(self.predictions_directory, 'Prophet_Test_Predictions.csv'), index=False)
            res_train.to_csv(os.path.join(self.predictions_directory, 'Prophet_Train_Predictions.csv'), index=False)

        if self.show_plots:
            plt.show()


# Main class, pass the parameters from the pipeline configuaration file
main = Forecaster(high_correlated_features=high_correlated_features, energy_json_directory=path_to_energy_data, power_json_directory=path_to_power_data, temp_humi_json_directory=path_to_temp_humi_data, weather_csv_directory=path_to_weather_data,
                  show_plots=show_plots, save_plots=save_plots, use_only_correct_data=use_only_correct_data, fix_if_sensor_restart = fix_if_sensor_restart, save_predictions = save_predictions, save_evaluation = save_evaluation,
                  good_time_start=good_time_start, good_time_end=good_time_end, start_time = start, scaler_f = scaler, site_number = site_number)

# Run the model specified in the config file with the parameters
if choose_model_to_run=="fb_prophet":                
    di = main.train_Prophet(test_size = prophet_test_size, only_energy = prophet_only_energy, add_holidays = add_holidays, period=period, fourier_order=fourier_order)

if choose_model_to_run=="sarimax":     
    di = main.train_SarimaX(test_size = sarimax_test_size, only_energy = sarimax_only_energy, exogeneous_data=exogeneous_data, grid_search_for_order_params=grid_search_for_order_params)

if choose_model_to_run=="arima":     
    di = main.train_Arima(test_size = arima_test_size, only_energy = arima_only_energy)

if choose_model_to_run=="regression":
    di = main.train_Regression(test_size = regression_test_size, only_energy=regression_only_energy)

if choose_model_to_run=="recurrent_neural_network":     
    di = main.train_Rnn(test_size = rnn_test_size, only_energy = rnn_only_energy, epochs= rnn_epochs, batch_size=rnn_batch_size, optimizer=rnn_optimizer)
