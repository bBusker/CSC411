import part2
import part3
from scipy.io import loadmat
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from constants import *


def grad_descent(f, df, x, y, init_W, alpha, _max_iter, momentum=0, printing=True):
    print("------------------------- Starting Grad Descent -------------------------")
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_W - 10 * EPS
    prev_grad = 0
    W = init_W.copy()
    max_iter = _max_iter
    iter = 0

    while iter < max_iter: #and norm(t - prev_t) > EPS:
        prev_t = W.copy()
        b = np.zeros(shape=(K_NUM,1))
        grad = df(x, y, W, b)
        W -= alpha * grad + momentum * prev_grad
        if iter % 100 == 0 and printing:
            print("Iter %i: cost = %.2f" % (iter,  f(x, W, b, y)))
        elif iter % 50000 == 0:
            print("Training...")
        iter += 1
        prev_grad = grad

    print("Done!")
    return W


def generate_sets(database, size):
    train_set = np.zeros((size, N_NUM))
    sol_set = np.zeros((size))

    for i in range(size):
        rand_dgt = np.random.random_integers(0,9)
        train_set[i] = database["train"+str(rand_dgt)][i]
        sol_set [i] = rand_dgt

    return train_set.T, sol_set

def alt_gen_set(database, scale):
    x = np.zeros(shape = (N_NUM, M_TRAIN))
    y = np.zeros(shape = (K_NUM, M_TRAIN))

    count = 0
    #Load our example
    for i in range(10):
        currSet = database["train" + str(i)].T / 255.0
        x[:, count: count + currSet.shape[1]] = currSet      
        y[i, count: count + currSet.shape[1]] = 1
        count += currSet.shape[1] 

    x = x[:, 0:M_TRAIN / scale]
    y = y[:, 0:M_TRAIN / scale]

    return x,y

def test(database, size, W, b):
    test_set = []
    correct = 0

    for i in range(size):
        rand_dgt = np.random.random_integers(0, 9)
        test_set += [database["test"+str(rand_dgt)][i]]
        guess = part2.forward(test_set[i], W, b)
        if guess == rand_dgt:
            correct += 1

    return correct/size


def part4(alpha, _max_iter, printing):
    M = loadmat("mnist_all.mat")

    results = []
    x = []

    #TODO: bias
    #for i in range(0, M_TRAIN, 100):
    W = np.zeros((784, 10))
    # train_set, sol_set = generate_sets(M, 10)
    train_set, sol_set = alt_gen_set(M, 1)
    W = grad_descent(part3.f, part3.df, train_set, sol_set, W, alpha, _max_iter, 0, printing)
    print(W)
    #results += [test(M, 20, W, np.zeros((10)))]
    #x += [0]


    #plt.plot(results, x)
