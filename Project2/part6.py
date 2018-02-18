import part2
from part3 import f
import part4 
from scipy.io import loadmat
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import cPickle as pickle

def part6a():
    W = cPickle.load(open("part5W.p", 'rb')) #load weights

    #picking 2 weights near the center, we will take the 14th pixel in the 11th row and the 14th pixel in the 17th
    #10 * 28 + 13 = 293
    #16 * 28 + 13 = 461 as our values of n, and any output may be chosen [0,9]

    M = loadmat("mnist_all.mat")
    x, y = part4.alt_gen_set(M, 1)
    b = np.ones(shape = (K_NUM,1))

    X1 = np.linspace(-1, 1, 100)
    X2 = np.linsapce(-1, 1, 100)

    res = np.zeros(shape = (100,100))
    max_deviation = 1
    for i in range(100):
        for j in range(100):
            W[293, 5] += X1[i] * max_deviation 
            W[461, 5] += X2[j] * max_deviation
            res[i+50, j+50] = f(x, W, b, y)
            
    return X1, X2, res
    

def part6():
    X1, X2, res = part6a()
    
    plt.contour(X1, X2, res)
    fig = plt.gcf()
    fig.savefig('part5fig.png')
    plt.show()
    