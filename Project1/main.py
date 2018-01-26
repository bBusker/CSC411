import get_data
import learner
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *

import os

actors = list(set([a.split("\n")[0] for a in open("./subset_actors.txt").readlines()]))
extensions = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

#get_data.get_data()
image_count = get_data.image_count("./cropped")
training_sets, validation_sets, test_sets = get_data.generate_sets(actors)

#Part 3: Steve Carell vs Alec Baldwin
#Steve Carell: 1
#Alec Baldwin: -1
x,y,thetas = learner.generate_xyt(training_sets["Steve Carell"] + training_sets["Alec Baldwin"], [1 for i in range(len(training_sets["Steve Carell"]))] + [-1 for i in range(len(training_sets["Alec Baldwin"]))])
thetas_p3 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.01, 100)

testactors_p3 = {key:test_sets[key] for key in ["Alec Baldwin", "Steve Carell"]}
testanswers_p3 = {"Alec Baldwin": -1, "Steve Carell": 1}
learner.test(testactors_p3, testanswers_p3, thetas_p3)

#Part 4: Showing Thetas
plt.imshow(thetas_p3[1:].reshape((32,32)))
plt.show()

#Part 5: Overfitting
#Male: 1
#Female: -1
training_set_all = []
labels_malefemale_all = []
for actor in actors:
    training_set_all += training_sets[actor]
    if actor in ["Steve Carell", "Alec Baldwin", "Bill Hader"]:
        labels_malefemale_all += [1 for i in range(len(training_sets[actor]))]
    elif actor in ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]:
        labels_malefemale_all += [-1 for i in range(len(training_sets[actor]))]

x,y,thetas = learner.generate_xyt(training_set_all, labels_malefemale_all)
thetas_p4 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.002, 10000)

testactors_p4 = {key:test_sets[key] for key in actors}
testanswers_p4 = {"Steve Carell": 1, "Alec Baldwin": 1, "Bill Hader": 1, "Lorraine Bracco": -1, "Peri Gilpin": -1, "Angie Harmon": -1}
learner.test(testactors_p4, testanswers_p4, thetas_p4)