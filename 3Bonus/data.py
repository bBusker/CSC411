import csv
import sys
import re

def clean(title):
    title = re.sub(r'"', '', title)  
    title = re.sub(r"'", '', title)
    title = re.sub('[^a-zA-Z \n \' \"]', ' ', title)
    title = title.lower()
    return title

def generateHeadlines():
    realHeadlines = []
    fakeHeadlines = []

    """----load kaggle fakes---"""
    # with open('fake.csv', mode='r') as infile:
    #     csv.field_size_limit(sys.maxsize)
    #     reader = csv.reader(infile)
    #     titleLocation = 4
    #     typeLocation = -1
    #     for row in reader:
    #         if row[typeLocation] in ['fake', 'bs', 'clickbait']:
    #             title = clean(row[titleLocation])
    #             if title not in ["", " "]:
    #                 fakeHeadlines.append(title)