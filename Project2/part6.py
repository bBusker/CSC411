import part2
from part3 import *
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

    res_a = np.zeros(shape = (100,100))
    max_deviation = 1
    for i in range(100):
        for j in range(100):
            W[w1, 5] += X1[i] * max_deviation 
            W[w2, 5] += X2[j] * max_deviation
            res[i+50, j+50] = f(x, W, b, y)
            
    return X1, X2, res
    
K = 10
w1 = 293
w2 = 461

def part6bcd():
    W = cPickle.load(open("part5W.p", 'rb')) #load weights
    x,y = alt_gen_set(M, 1)

    W[w1, 5], W[w2, 5] = 0,0
    W, res_b = grad_descent_2element(f, df, x, y, W, 1, , 0, True)

    W[w1, 5], W[w2, 5] = 0,0
    W, res_c = grad_descent_2element(f, df, x, y, W, 1, , 0.95, True)

    return res_b, res_c



def grad_descent_2element(f, df, x, y, init_W, alpha, _max_iter, momentum=0, printing=True):
    print("------------------------- Starting Grad Descent -------------------------")
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_W - 10 * EPS
    prev_grad = 0
    W = init_W.copy()
    V = np.zeros(shape = W.shape)
    max_iter = _max_iter
    iter = 0

    res = np.zeros(shape = (1, K))
    while iter < max_iter: #and norm(t - prev_t) > EPS:
        prev_t = W.copy()
        res[iter] = (W[w1, 5], W[w2, 5])
        b = np.zeros(shape=(K_NUM,1))
        grad = df(x, y, W, b)
        V = momentum * V + alpha * grad
        # W -= alpha * grad
        W[w1, 5] -= V[w1, 5]
        W[w2, 5] -= V[w2, 5]
        if iter % 5000 == 0 and printing:
            print("Iter %i: cost = %.2f" % (iter, f(x, y, W, x.shape[1])))
        elif iter % 50000 == 0:
            print("Training...")
        iter += 1
        prev_grad = grad

    print("Done!")
    return W, res

def part6():
    X1, X2, res = part6a()
    a_weights, b_weights = part6bcd()
    
    plt.contour(X1, X2, res)
    plt.plot([a for a, b in a_weights], [b for a,b in a_weights], 'yo-', label="No Momentum")
    plt.plot([a for a, b in b_weights], [b for a,b in b_weights], 'go-', label="Momentum")
    plt.legend(loc='top left')
    fig = plt.gcf()
    fig.savefig('part5fig.png')
    plt.show()
    