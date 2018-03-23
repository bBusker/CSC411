import csv
import sys
import re

def loadHeadlines():
    realHeadlines = []
    fakeHeadlines = []

    #load kaggle fakes
    with open('fake.csv', mode='r') as infile:
        csv.field_size_limit(sys.maxsize)
        reader = csv.reader(infile)
        titleLocation = 4
        typeLocation = -1
        for row in reader:
            if row[typeLocation] in ['fake', 'bs', 'clickbait']:
                title = row[titleLocation]
                print(title)
                title = re.sub('[^a-zA-Z0-9 \n\.]', '', title)
                title = re.sub('\d+', "", title)
                print(title)
                if title not in ["", " "]:
                    fakeHeadlines.append(title)
    # print(fakeHeadlines[0:100])