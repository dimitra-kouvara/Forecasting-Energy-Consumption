"""
PROJECT: FORECASTING RESIDENTIAL CONSUMPTION
Weather Scraping

Site - weather operator
https://www.accuweather.com/

@author: Dimitra
"""

# importing the necessary packages
import http_connection
import web_scraper

if __name__ == "__main__": 
    
    proxy = http_connection.get_proxy()
    agents = http_connection.get_agent()
    
    # Weather operator's URL
    url = 'https://www.accuweather.com/en/gr/athens/'
    link_prefix = 'https://www.accuweather.com'
    
    suffixes = web_scraper.get_links(url, agents, proxy)
    today_suffix, hourly_suffix, daily_suffix, monthly_suffix = suffixes[0], suffixes[1], suffixes[2], suffixes[3]
                
    # get the links of the three next months
    # nextMonth_links = web_scraper.get_nextMonthLinks(link_prefix, monthly_suffix, agents, proxy)
    # get a json file with the lowest and highest temperatures per date for the three next months
    # monthlyTemps = web_scraper.get_monthTemp(nextMonth_links, agents, proxy)
    
    # get the data (low and high temperatures) for all months - past and upcoming
    allMonth_links = web_scraper.get_allMonthLinks(link_prefix, monthly_suffix) 
    
    # get a json file with the lowest and highest temperatures per date for all months -  past and upcoming
    monthlyTemps = web_scraper.get_monthTemp(allMonth_links, agents, proxy)
    

    # get humidity from api
    # check March
    # To do:
                            # save to SQL database
