"""
PROJECT: FORECASTING RESIDENTIAL CONSUMPTION
Weather Scraping

Site - weather operator
https://www.accuweather.com/

@author: Dimitra
"""

# find clusters of numbers in a list
def grouper(iterable):
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev == 1:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


# find the valid dates of month, corresponding to the longest list 
def find_valid_dates(values):
    list_len = [len(i) for i in values]
    max_list = max(list_len)
    max_index = list_len.index(max_list)
    return max_list, max_index # position of longest list in calendar