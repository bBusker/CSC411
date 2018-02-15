from pylab import *
from scipy.io import loadmat


# PART 1

def part1():
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    # save 10 images, 1 from each digit
    for i in range(10):
        imsave("part1/image" + str(i) + ".png", M["train"+str(i)][10].reshape((28,28)))
