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

tempactors = ['Peri Gilpin']
def get_data():
    for a in tempactors: #TESTING
        name = a.split()[1].lower()
        i = 0
        for line in open("./faces_subset.txt"):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
                retdata = timeout(urllib.request.urlretrieve, (line.split()[4], "./uncropped/" + filename), {}, 2)
                if not os.path.isfile("./uncropped/" + filename) or retdata == False or retdata == None:
                    continue
                try:
                    imarr = imread(retdata[0], True)
                    coords = line.split()[5]
                    x1 = int(coords.split(',')[0])
                    y1 = int(coords.split(',')[1])
                    x2 = int(coords.split(',')[2])
                    y2 = int(coords.split(',')[3])
                    imcropped = imarr[y1:y2, x1:x2]
                    imsave("./cropped/" + filename, imcropped)
                    print(filename)
                    i += 1

                except:
                    print("invalid image")
                    os.remove(retdata[0])

