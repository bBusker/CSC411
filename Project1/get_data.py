from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import *
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib.request

actors = list(set([a.split("\n")[0] for a in open("./subset_actors.txt").readlines()]))


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        print("timeout")
        return False
    else:
        return it.result


def get_data():
    for a in actors:
        name = a.split()[1].lower()
        i = 0
        for line in open("./faces_subset.txt"):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
                # if os.path.isfile("./uncropped/" + filename): #doesnt account for different extensions
                #     i += 1
                #     print(filename + " exists")
                #     continue
                retdata = timeout(urllib.request.urlretrieve, (line.split()[4], "./uncropped/" + filename), {}, 5)
                if not os.path.isfile("./uncropped/" + filename) or retdata == False or retdata == None:
                    continue
                try:
                    imarr = imread(retdata[0], True)
                    coords = line.split()[5]
                    x1,y1,x2,y2 = list(map(int,(coords.split(','))))
                    imcropped = imarr[y1:y2, x1:x2]
                    imfinal = imresize(imcropped, (32,32))
                    imsave("./cropped/" + filename, imfinal)
                    print(filename)
                    i += 1

                except:
                    print("invalid image")
                    os.remove(retdata[0])


def image_count(path):
    res = {key: 0 for key in actors}
    for file in os.listdir(path):
        for actor in actors:
            if file.startswith(actor.split()[1].lower()):
                res[actor] += 1
    return res
