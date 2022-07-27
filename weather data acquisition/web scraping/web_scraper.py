"""
PROJECT: FORECASTING RESIDENTIAL CONSUMPTION
Weather Scraping

Site - weather operator
https://www.accuweather.com/

@author: Dimitra
"""
# importing the necessary packages
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import random
import re
import json
import valid_days
import datetime
import time
import calendar



# initializing variables         
# path to write the files
path = ''
json_extention = '.json'
txt_extention = '.txt'
monthLinks = list()
list_links = list()
allMonth_links = list()
output = dict()  
output["date"], output["temp_low"], output["temp_high"] = list(), list(), list()


# ==========================================================================================================
#                                                Functions

# use it to serialize the datetime into json
def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

def get_allMonthLinks(link_prefix, monthly_suffix):
    global allMonth_links
    # get a list of the months
    months = calendar.month_name[1:]
    lowercaseMonths = [each_month.lower() for each_month in months]
    # get the current, previous, following year, as Accuweather has info for all of these
    current_monthNo = datetime.datetime.now().month
    current_month = calendar.month_name[current_monthNo].lower()
    current_year = datetime.datetime.now().year
    previous_year = current_year - 1
    next_year = current_year + 1
    years = [str(current_year), str(previous_year), str(next_year)]
    for year in years:
        for month in lowercaseMonths:
            month_suffix = monthly_suffix.replace(current_month, month)
            if year != str(current_year):
                year_suffix = "".join(['?year=', year])
            else:
                year_suffix = ""
            changedMonth_link = "".join([link_prefix, month_suffix, year_suffix])
            allMonth_links.append(changedMonth_link)
    return allMonth_links


def get_links(url, agents, proxy):
    try:
        # send the request to server
        page = requests.get(url, headers = agents, proxies = proxy, timeout = random.randint(2,4))
        coverpage = page.content
        soup = BeautifulSoup(coverpage, 'html.parser')
        
        # Weather forecasting acquisition
        body = soup.find('body').find('div', class_="template-root").find('div', class_="two-column-page-content").find('div', class_="page-column-1")
        pageContent = body.find('div', class_="content-module").find('div', class_="more-cta-links content-module")
        links = pageContent.find_all('a')
        
        for link in links:
            if link.find(text=re.compile("Today")):
                today_suffix = link.get('href')
            elif link.find(text=re.compile("Hourly")):
                hourly_suffix = link.get('href')
            elif link.find(text=re.compile("Daily")):    
                daily_suffix = link.get('href')
            elif link.find(text=re.compile("Monthly")):
                monthly_suffix = link.get('href')
            else:
                pass
            
        time.sleep(random.randint(3,6))
        return today_suffix, hourly_suffix, daily_suffix, monthly_suffix 
    
    except ConnectionError:
        print("Connection refused (central page loading)")
    
    
# get the links of the three next months
def get_nextMonthLinks(link_prefix, link_suffix, agents, proxy):
    global monthLinks
    link = "".join([link_prefix, link_suffix])
    monthLinks.append(link)

    try:
        article = requests.get(link, headers = agents, proxies = proxy, timeout = random.randint(2,4))
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html.parser')
        
        try:
            body = soup_article.find('body').find('div', class_="template-root").find('div', class_="two-column-page-content").find('div', class_="page-column-1")
            pageContent = body.find('div', class_="content-module")
            
            # Finding the url for the next months
            nextMonths_pattern = pageContent.find('div', class_="more-cta-links")
            nextMonths = nextMonths_pattern.find_all('a', class_="cta-link")
            
            for month in nextMonths:
                monthLink_suffix = month.attrs['href']
                monthLink = " ".join([link_prefix, monthLink_suffix])
                monthLinks.append(monthLink)
                
        except:
            print("The server is overloaded, and no more links can be established")
            print(" Reasong: The IP is blocked by the target website due to the high request frequency")
            pass
        
        time.sleep(random.randint(3,6))
        return monthLinks
    
    except requests.exceptions.ConnectionError:
        print("Connection refused")
                    
        
# get the low and high temperatures per date for a specific month    
def scrape_monthTemp(month_link, agents, proxy, list_days, list_dates, list_tempHigh, list_tempLow):
    global txt_extention
    try:
        article = requests.get(month_link, headers = agents, proxies = proxy, timeout = random.randint(2,4))
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html.parser')
        
        try:
            body = soup_article.find('body').find('div', class_="template-root").find('div', class_="two-column-page-content").find('div', class_="page-column-1")
            pageContent = body.find('div', class_="content-module")
            
            monthYear_pattern = pageContent.find('div', class_="monthly-tools non-ad").find('div', class_="monthly-dropdowns")
            monthYear = monthYear_pattern.find_all('div', class_="map-dropdown") 
            
            # Current month and year
            month = monthYear[0].find('div', class_="map-dropdown-toggle").get_text().strip('\n')
            year = monthYear[1].find('div', class_="map-dropdown-toggle").get_text().strip('\n')
            
            calendarContent = pageContent.find('div', class_="monthly-component non-ad").find('div', class_="monthly-calendar-container")
            
            days_pattern = calendarContent.find('div', class_="monthly-header")
            days = days_pattern.find_all('div', class_="day-text")  
            for day in days:
                # Day (S-M)
                shortDay = day.get_text().replace('\n','').replace('\t','')
                list_days.append(shortDay)
                
            dates_pattern = calendarContent.find('div', class_="monthly-calendar").find_all('a')
            
            for pattern in dates_pattern:
                attributeContent = pattern.find('div', class_="monthly-panel-top")
                # Date (1-31)
                date = int(attributeContent.find('div', class_="date").get_text().replace('\n','').replace('\t',''))
                list_dates.append(date)

            txtName = " ".join(["weather", year, month])
            txtFile = "".join([path, txtName, txt_extention])    
            with open(txtFile, "w") as f:
                for s in dates_pattern:
                    f.write(str(s) +"\n")
                    
            # Path.read_text() function opens the file in text mode, reads it, and close the file
            weather = Path(txtFile).read_text()
                      
        
            weather_list = re.split('</a>', weather)
            temp_pattern = '[0-9]+°'
            
            for ele in weather_list:
                temp = re.findall(temp_pattern, ele)
                if len(temp) != 0: 
                    temp_high = temp[0].strip('°')
                    temp_low = temp[1].strip('°')
                else:
                    temp_high = ''
                    temp_low = ''
                list_tempHigh.append(temp_high)
                list_tempLow.append(temp_low)
        except: # Max retries exceeded with url
            print("The server is overloaded, and no more links can be established")
            print(" Reasong: The IP is blocked by the target website due to the high request frequency")
            pass   
        return year, month
    except ConnectionError:
        print("Connection refused (forecasting)")


# get the low and high temperatures per date for a specific month    
def get_monthTemp(month_links, agents, proxy):
    global output, json_extention, path
    
    try:
        for month_link in month_links:
            # initialization of lists
            list_dates = []
            list_days = []
            list_tempLow = []
            list_tempHigh = []
            
            forecast = scrape_monthTemp(month_link, agents, proxy, list_days, list_dates, list_tempHigh, list_tempLow)
            year = forecast[0]
            month = forecast[1]
            
            calendar = dict(enumerate(valid_days.grouper(list_dates), 1))
            max_key = max(calendar, key = lambda x: len(set(calendar[x])))
            month_dates = calendar[max_key]
            month_dates = list(map(str, month_dates))
            
            # Starting and ending position of dates (of current month) in calendar                    
            startEnd_position = valid_days.find_valid_dates(calendar.values())
            list_range = startEnd_position[0]  
            index = startEnd_position[1]
            
            if index != 0:
                start_position = len(calendar[index])
            else:
                start_position = index + 1
                
            end_position = start_position + list_range-1   
                
            
            complete_date = list()
            for i in range(0, len(month_dates)):
                date = " ".join([month_dates[i], month, year])
                complete_date.append(date)
            
            # 
            complete_tempLow = list_tempLow[start_position: end_position+1]
            complete_tempHigh = list_tempHigh[start_position: end_position+1]
            
            for i in range(0, len(complete_date)):
                output['date'].append(complete_date[i])
                output["temp_low"].append(complete_tempLow[i])
                output["temp_high"].append(complete_tempHigh[i])
            
            # serializing json
            jsonObject = json.dumps(output, indent = 4, default=myconverter)
            
            # writing to json
            jsonName = "monthly forecast"
            jsonFile = "".join([path, jsonName, json_extention])
            
            with open(jsonFile, "w") as outfile:
              outfile.write(jsonObject)
     
    except TypeError:
        print("Failed due to connection refusal")