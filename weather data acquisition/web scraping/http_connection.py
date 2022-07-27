"""
PROJECT: FORECASTING RESIDENTIAL CONSUMPTION
Weather Scraping

Site - weather operator
https://www.accuweather.com/

@author: Dimitra
"""
import random 

# pass in to requests some random user agent to represent a human and different machine for the STARTING PAGE
def chromeUserStartingPage(agentsList):
    credentialsDict = {      
                        'User-Agent': random.choice(agentsList),
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                        'Upgrade-Insecure-Requests': '1'}
    
    return credentialsDict


# pass in to requests some random user agent to represent a human and different machine
def chromeUserGenerator(agentsList):
    credentialsDict = {      
                        'User-Agent': random.choice(agentsList),
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                        'Accept-Language': 'el-GR,el;q=0.9,en;q=0.8',
                        'Upgrade-Insecure-Requests': '1'}
    
    return credentialsDict


# use http protocol free proxies to attemp to scrape
def randomProxyGenerator(proxyList):
    randomDict = {'http': 'http://'+random.choice(proxyList)}
    
    return randomDict
  

def get_proxy():
    # import a file and create the random proxies
    proxiesListFile = 'proxiesList.txt'
    with open(proxiesListFile, 'r') as inputFile:
        proxiesList = inputFile.readlines()
        
    for item in range(len(proxiesList)):
        proxiesList[item] = proxiesList[item][:-1]
        
    return randomProxyGenerator(proxiesList)


def get_agent():
    return {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64)'}