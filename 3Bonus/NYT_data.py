import json
import nytimes
import time
import csv
import re

search_obj = nytimes.get_article_search_obj ('d66d7d7ec5264a9491a9a32cd3960d9b')
data = []

for x in range(200):
    try:
        print ("Page %d" %(x))
        f = search_obj.article_search(q='trump', fl=['headline'], begin_date='20180301', page=str(x), sort='newest')   
    except:break

    try:
        # print f['response']
        for k in f['response']['docs']:
            title = k['headline']['print_headline'].encode('utf-8')
            if title:
                data.append([title])
        # print (data)
        time.sleep(1)
    except:
        pass

print ("Writing to file...")
with open('ny_times.csv', 'w') as f:
    wr = csv.writer(f)
    wr.writerows(data)


 