import csv
import sys
import re
import json
# import nytimes
import time
import requests
import os
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

def clean(title):
    title = re.sub('[\'\"\.]', "", title)
    title = re.sub('[^a-zA-Z \n \' \ \."]', ' ', title)
    title = title.lower()
    title = re.sub(' +',' ',title)
    return title

def generateHeadlines():
    realHeadlines = []
    fakeHeadlines = []

    """----load kaggle fakes---"""
    with open('fake.csv', mode='r') as infile:
        csv.field_size_limit(sys.maxsize)
        reader = csv.reader(infile)
        titleLocation = 4
        typeLocation = -1
        for row in reader:
            if row[typeLocation] in ['fake', 'bs']:
                title = clean(row[titleLocation])
                if title not in ["", " "]:
                    fakeHeadlines.append(title)

    """----load McIntyre's set---"""
    with open("fake_or_real_news.csv", mode = 'r') as infile:
        csv.field_size_limit(sys.maxsize)
        reader = csv.reader(infile)
        titleLocation = 1
        typeLocation = 3
        for row in reader:
            title = clean(row[titleLocation])
            if title not in ["", " "]:
                if row[typeLocation] == "FAKE":
                    fakeHeadlines.append(title)
                elif row[typeLocation] == "REAL":
                    realHeadlines.append(title)

    """----load Kaggle set #2---"""
    with open("kaggledata2.csv", mode = 'r') as infile:
        csv.field_size_limit(sys.maxsize)
        reader = csv.reader(infile)
        titleLocation = 1
        typeLocation = 3
        for row in reader:
            title = clean(row[titleLocation])
            if title not in ["", " "]:
                if row[typeLocation] == "0":
                    fakeHeadlines.append(title)
                elif row[typeLocation] == "1":
                    realHeadlines.append(title)

    print("beginning guardian")
    ARTICLES_DIR = join('tempdata', 'articles')
    makedirs(ARTICLES_DIR, exist_ok=True)
    MY_API_KEY = "b8871c67-9291-40a1-9b04-f0933d4862e3"
    API_ENDPOINT = 'http://content.guardianapis.com/search'
    my_params = {
        'from-date': "",
        'to-date': "",
        'order-by': "newest",
        'show-fields': 'all',
        'page-size': 200,
        'api-key': MY_API_KEY
    }
    start_date = date(2015, 3, 1)
    # start_date = date(2018, 3, 1)
    end_date = date(2018,3, 23)
    dayrange = range(0, (end_date - start_date).days, 30)
    for daycount in dayrange:
        dt = start_date + timedelta(days=daycount)
        datestr = dt.strftime('%Y-%m-%d')
        fname = join(ARTICLES_DIR, datestr + '.json')
        if not exists(fname):
            # then let's download it
            all_results = []
            my_params['from-date'] = datestr
            my_params['to-date'] = datestr
            current_page = 1
            total_pages = 1
            while current_page <= total_pages:
                my_params['page'] = current_page
                resp = requests.get(API_ENDPOINT, my_params)
                data = resp.json()
                for result in data['response']['results']:
                    realHeadlines.append(clean(result['fields']['headline']))
                current_page += 1
                total_pages = data['response']['pages']

    print("total size: " + str(len(realHeadlines) + len(fakeHeadlines)))
    print("fake size: " + str(len(fakeHeadlines)))
    print("real size: " + str(len(realHeadlines)))

    realFile = open("realHeadlines.txt", "a")
    fakeFile = open("fakeHeadlines.txt", "a")

    for headline in realHeadlines:
        realFile.write(headline + "\n")

    for headline in fakeHeadlines:
        fakeFile.write(headline + "\n")



