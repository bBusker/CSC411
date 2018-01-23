import get_data
import learner
import numpy as np
from scipy.misc import *

import os

actors = list(set([a.split("\n")[0] for a in open("./subset_actors.txt").readlines()]))
extensions = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

#get_data.get_data()
image_count = get_data.image_count("./cropped")
training_sets, validation_sets, test_sets = get_data.generate_sets(actors)

#Part 3: Alec Baldwin vs Steve Carell
thetas = np.zeros((1025,1))
x_carell = np.zeros((image_count["Steve Carell"] - 20, 1025))
x_baldwin = np.zeros((image_count["Alec Baldwin"] - 20, 1025))
y = np.zeros((x_carell.shape[0] + x_baldwin.shape[0], 1))

j = 0
for actor in ["Steve Carell", "Alec Baldwin"]:
    i = 0
    for image in training_sets[actor]:
        imdata = (imread("./cropped/" + image[1])/255).reshape(1024)
        imdata = np.concatenate(([1], imdata))
        if actor == "Steve Carell":
            x_carell[i] = imdata
            y[j] = 1
        elif actor == "Alec Baldwin":
            x_baldwin[i] = imdata
            y[j] = -1
        i += 1
        j += 1

x = np.vstack((x_carell, x_baldwin))
x = np.transpose(x)
thetas = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.00001)
