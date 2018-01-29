import get_data
import learner
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *

import os

# Getting and processing data
actors = list(set([a.split("\n")[0] for a in open("./subset_actors.txt").readlines()]))
actors_orig = ["Steve Carell", "Alec Baldwin", "Bill Hader", "Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]
actors_new = ["Michael Vartan", "Gerard Butler", "Daniel Radcliffe", "Kristin Chenoweth", "America Ferrera", "Fran Drescher"]
extensions = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

# get_data.get_data()
image_count = get_data.image_count("./cropped")
training_sets, validation_sets, test_sets = get_data.generate_sets(actors)


# Part 3: Steve Carell vs Alec Baldwin
# Steve Carell: 1
# Alec Baldwin: -1
x,y,thetas = learner.generate_xyt(training_sets["Steve Carell"] + training_sets["Alec Baldwin"], [1 for i in range(len(training_sets["Steve Carell"]))] + [-1 for i in range(len(training_sets["Alec Baldwin"]))])
thetas_p3 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.01, 100)

testactors_p3 = {key:test_sets[key] for key in ["Alec Baldwin", "Steve Carell"]}
testanswers_p3 = {"Alec Baldwin": -1, "Steve Carell": 1}
learner.test(testactors_p3, testanswers_p3, thetas_p3)


# Part 4: Showing Thetas
plt.imshow(thetas_p3[1:].reshape((32,32)))
plt.show()


# Part 5: Overfitting
# Male: 1
# Female: -1
training_set_orig_6 = []
labels_malefemale_all = []

for actor in actors_orig:
    training_set_orig_6 += training_sets[actor]
    if actor in ["Steve Carell", "Alec Baldwin", "Bill Hader"]:
        labels_malefemale_all += [1 for i in range(len(training_sets[actor]))]
    elif actor in ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]:
        labels_malefemale_all += [-1 for i in range(len(training_sets[actor]))]

x,y,thetas = learner.generate_xyt(training_set_orig_6, labels_malefemale_all)
thetas_p4 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.002, 200000)

testactors_p4a = {key:test_sets[key] for key in actors_orig}
testanswers_p4a = {"Steve Carell": 1, "Alec Baldwin": 1, "Bill Hader": 1, "Lorraine Bracco": -1, "Peri Gilpin": -1, "Angie Harmon": -1}
learner.test(testactors_p4a, testanswers_p4a, thetas_p4)

testactors_p4b = {key:test_sets[key] for key in actors_new}
testanswers_p4b = {"Michael Vartan": 1, "Gerard Butler": 1, "Daniel Radcliffe": 1, "Kristin Chenoweth": -1, "America Ferrera": -1, "Fran Drescher": -1}
learner.test(testactors_p4b, testanswers_p4b, thetas_p4)
# TODO: 6 diff actors