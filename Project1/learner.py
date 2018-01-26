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

#Modified Guerzhoy's
def quad_loss(x, y, theta):
    #x = vstack((ones((1, x.shape[1])), x))
    return sum((y.T - dot(theta.T, x)) ** 2)


def quad_loss_grad(x, y, theta, norm_const):
    #x = vstack((ones((1, x.shape[1])), x))
    return -2 * sum((y.T - dot(theta.T, x)) * x, 1) / norm_const


def generate_xyt(input_sets, labels):
    x = np.zeros((len(input_sets), 1025))
    y = np.zeros((len(input_sets),1))
    thetas = zeros((1025,1))
    for i in range(len(input_sets)):
        imdata = (imread("./cropped/" + input_sets[i][1]) / 255).reshape(1024)
        imdata = np.concatenate(([1], imdata))
        x[i] = imdata
        y[i] = labels[i]
    return x.T, y, thetas


def grad_descent(f, df, x, y, init_t, alpha, _max_iter):
    EPS = 1e-5  #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = _max_iter
    iter = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        grad = df(x, y, t, t.shape[0]).reshape(1025,1)
        t -= alpha*(grad)
        if iter % 1000 == 0:
            print("Iter %i: cost = %.2f" % (iter, f(x, y, t)))
            #print("Gradient: ", grad, "\n")
        iter += 1
    print("------------------------- Finished Grad Descent -------------------------")
    print("")
    return t


def test(test_sets, answers, thetas):
    correct = 0
    count = 0
    print("------------------------- Testing -------------------------")
    for actor in test_sets:
        i = 0
        for image in test_sets[actor]:
            imdata = (imread("./cropped/" + image[1]) / 255).reshape(1024)
            imdata = np.concatenate(([1], imdata))
            prediction = np.dot(imdata, thetas)[0]
            print("%s %i|pred: %.2f, ans: %.2f" % (actor, i, prediction, answers[actor]))
            if abs(answers[actor] - prediction) < 1:
                correct += 1
            # min = 999
            # guess = 0
            # for answer in answers:
            #     if abs(prediction - answers[answer]) < min:
            #         min = abs(prediction - answers[answer])
            #         guess = answer
            # if guess == image[0]:
            #     correct += 1
            i += 1
            count += 1
    print("Score: %.2f" % (correct / count))
