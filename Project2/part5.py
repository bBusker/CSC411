import part2
import part3
import part4
from scipy.io import loadmat
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import cPickle as pickle

def grad_descent(f, df, x, y, init_W, alpha_w, alpha_b, _max_iter, momentum = 0, printing=True):
    print("------------------------- Starting Grad Descent -------------------------")
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_W - 10 * EPS
    W = init_W.copy()
    b = np.zeros(shape=(K_NUM, 1))
    V_w = np.zeros(shape = W.shape)
    V_b = np.zeros(shape = b.shape)
    max_iter = _max_iter
    iter = 0

    while iter < max_iter: #and norm(t - prev_t) > EPS:
        #prev_t = W.copy()
        grad_w, grad_b = df(x, y, W, b)
        V_w = momentum * V_w + alpha_w * grad_w
        W -= V_w
        V_b = momentum * V_b + alpha_b * grad_b
        b -= V_b
        if iter % 100 == 0 and printing:
            print("Iter %i: cost = %.5f" % (iter,  f(x, W, b, y)))
        elif iter % 50000 == 0:
            print("Training...")
        iter += 1

    print("Done!")
    return W, b


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

def test2(database, solutions, W, b):
    correct = 0
    database = database.T
    solutions = solutions.T

    for i in range(len(database)):
        guess = part2.forward(database[i], W, b)
        if np.argmax(guess) == np.argmax(solutions[i]):
            correct += 1

    return correct/float(len(database))


def part5(alpha_w, alpha_b, _max_iter, printing):
    M = loadmat("mnist_all.mat")

    results_val = []
    results_train = []
    results_test = []
    x = []
    train_set, train_set_sol, val_set, val_set_sol, test_set, test_set_sol = part4.generate_sets(M, 5000)
    W_init = np.random.rand(784, 10)

    step = 20
    for i in range(step, _max_iter, step):
        print(i)
        W, b = grad_descent(part3.f, part3.df, train_set, train_set_sol, W_init, alpha_w, alpha_b, i, 0.9, True)

        y_pred = part2.forward(val_set, W, b)
        results_val += [np.mean(np.argmax(y_pred, 0) == np.argmax(val_set_sol, 0))]
        y_pred = part2.forward(train_set, W, b)
        results_train += [np.mean(np.argmax(y_pred, 0) == np.argmax(train_set_sol, 0))]
        y_pred = part2.forward(test_set, W, b)
        results_test += [np.mean(np.argmax(y_pred, 0) == np.argmax(test_set_sol, 0))]
        print results_test 
        print results_train
        print results_val
        x += [i]

    pickle.dump(W, open("part4W_shichen.p", "wb"))
    plt.plot(x, results_val,'y',label="validation set")
    plt.plot(x, results_train,'g',label="training set")
    plt.plot(x, results_test,'r',label="test set")
    plt.title("Part 5 Learning Curve")
    plt.ylabel("accuracy")
    plt.xlabel("iterations")
    plt.show()