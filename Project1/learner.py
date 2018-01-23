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


#Guerzhoy's
def quad_loss(x, y, theta):
    #x = vstack((ones((1, x.shape[1])), x))
    return sum((y.T - dot(theta.T, x)) ** 2)


def quad_loss_grad(x, y, theta, norm_const):
    #x = vstack((ones((1, x.shape[1])), x))
    return -2 * sum((y.T - dot(theta.T, x)) * x, 1) / norm_const


def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5  #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 100000
    iter = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        grad = df(x, y, t, t.shape[0]).reshape(1025,1)
        t -= alpha*(grad)
        if iter % 1000 == 0:
            print("Iter", iter)
            print("x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)))
            #print("Gradient: ", grad, "\n")
        iter += 1
    return t


# def quad_loss(thetas, image_set, label0, label1):
#     res = 0
#     i = 0
#     for image in image_set:
#         i += 1
#         imdata = (imread("./cropped/" + image[1])/255).reshape(1024)
#         imdata = np.concatenate(([1], imdata))
#         if(image[0] == label0):
#             res += (np.dot(thetas, imdata)[0])^2
#         elif(image[0] == label1):
#             res += (np.dot(thetas, imdata)[0] - 1)^2 #TODO: test quad cost function
#     return res/(2*i)
#
# def quad_loss_grad(thetas, image_set, label0, label1):
#     res = zeros(1, 1025)
#     for image in image_set:
#

# def grad_desc(cost_func, train_set, val_set, test_set):
#     thetas = np.zeros((1, 1024))
#
#     while(exp_loss())
