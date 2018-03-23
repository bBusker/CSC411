import csv
import sys
import re

def clean(title):
    title = title.replace('"', "")
    title = title.replace("'", "")
    title = re.sub('[^a-zA-Z \n \' \"]', ' ', title)
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
    print(fakeHeadlines[0:100])