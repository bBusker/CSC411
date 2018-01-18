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


def sigmoid(z):
    return 1/(1+exp(-z))


def exp_loss(thetas, image_set, label0, label1):
    res = 0
    i = 0
    for image in image_set:
        i += 1
        sig = sigmoid(np.dot(thetas, (imread("./cropped/" + image[1])/255).reshape(1024))[0])
        if(image[0] == label0):
            res += -log(sig)
        elif (image[0] == label1):
            res += -log(1-sig) #TODO: divide by 0 bug
    return res/(2*i)


def exp_loss_grad(y, z, x):
    sig = sigmoid(z)
    if(y == 1):
        return (1 - sig)*(x)
    elif(y == 0):
        return (sig)*(x)

# def grad_desc(cost_func, train_set, val_set, test_set):
#     thetas = np.zeros((1, 1024))
#
#     while(exp_loss())
