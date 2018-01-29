import get_data
import learner
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *

import os

# Getting and processing data
actors = list(set([a.split("\n")[0] for a in open("./subset_actors.txt").readlines()]))
actors_orig = ["Alec Baldwin", "Steve Carell", "Bill Hader", "Lorraine Bracco", "Angie Harmon", "Peri Gilpin"]
actors_new = ["Michael Vartan", "Gerard Butler", "Daniel Radcliffe", "Kristin Chenoweth", "America Ferrera", "Fran Drescher"]
extensions = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

# get_data.get_data()
image_count = get_data.image_count("./cropped")
training_sets, validation_sets, test_sets = get_data.generate_sets(actors)


# Part 3: Steve Carell vs Alec Baldwin
# Steve Carell: 1
# Alec Baldwin: -1
print("\n\n >>>PART 3<<<")

x,y,thetas = learner.generate_xyt(training_sets["Steve Carell"] + training_sets["Alec Baldwin"], [1 for i in range(len(training_sets["Steve Carell"]))] + [-1 for i in range(len(training_sets["Alec Baldwin"]))])
thetas_p3 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.001, 10000)

testactors_p3 = {key:test_sets[key] for key in ["Alec Baldwin", "Steve Carell"]}
testanswers_p3 = {"Alec Baldwin": np.array((-1)), "Steve Carell": np.array((1))}
learner.test(testactors_p3, testanswers_p3, thetas_p3)


# Part 4: Showing Thetas
plt.imshow(thetas_p3[1:].reshape((32,32)))
plt.title("Part 3")
plt.show()


# Part 5: Overfitting
# Male: 1
# Female: -1
print("\n\n >>>PART 5<<<")

training_set_orig_6 = []
labels_malefemale_all = []

for actor in actors_orig:
    training_set_orig_6 += training_sets[actor]
    if actor in ["Steve Carell", "Alec Baldwin", "Bill Hader"]:
        labels_malefemale_all += [1 for i in range(len(training_sets[actor]))]
    elif actor in ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]:
        labels_malefemale_all += [-1 for i in range(len(training_sets[actor]))]

x,y,thetas = learner.generate_xyt(training_set_orig_6, labels_malefemale_all)
thetas_p4 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.001, 10000)

testactors_p4a = {key:test_sets[key] for key in actors_orig}
testanswers_p4a = {"Steve Carell": np.array((1)), "Alec Baldwin": np.array((1)), "Bill Hader": np.array((1)), "Lorraine Bracco": np.array((-1)), "Peri Gilpin": np.array((-1)), "Angie Harmon": np.array((-1))}
learner.test(testactors_p4a, testanswers_p4a, thetas_p4)

testactors_p4b = {key:test_sets[key] for key in actors_new}
testanswers_p4b = {"Michael Vartan": np.array((1)), "Gerard Butler": np.array((1)), "Daniel Radcliffe": np.array((1)), "Kristin Chenoweth": np.array((-1)), "America Ferrera": np.array((-1)), "Fran Drescher": np.array((-1))}
learner.test(testactors_p4b, testanswers_p4b, thetas_p4)


# Part 7: Multiple Actor Classification
# Alec Baldwin:    [1,0,0,0,0,0]
# Steve Carell:    [0,1,0,0,0,0]
# Bill Hader:      [0,0,1,0,0,0]
# Lorraine Bracco: [0,0,0,1,0,0]
# Angie Harmon:    [0,0,0,0,1,0]
# Peri Gilpin:     [0,0,0,0,0,1]
print("\n\n >>>PART 7<<<")

training_set_orig_6 = []
labels = []

for actor in actors_orig:
    training_set_orig_6 += training_sets[actor]
    if actor == "Alec Baldwin":
        labels += [[1,0,0,0,0,0] for i in range(len(training_sets[actor]))]
    elif actor == "Steve Carell":
        labels += [[0,1,0,0,0,0] for i in range(len(training_sets[actor]))]
    elif actor == "Bill Hader":
        labels += [[0,0,1,0,0,0] for i in range(len(training_sets[actor]))]
    elif actor == "Lorraine Bracco":
        labels += [[0,0,0,1,0,0] for i in range(len(training_sets[actor]))]
    elif actor == "Angie Harmon":
        labels += [[0,0,0,0,1,0] for i in range(len(training_sets[actor]))]
    elif actor == "Peri Gilpin":
        labels += [[0,0,0,0,0,1] for i in range(len(training_sets[actor]))]

x,y,thetas = learner.generate_xyt(training_set_orig_6, labels)
thetas_p7 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.002, 10000)

testactors_p7 = {key:test_sets[key] for key in actors_orig}
testanswers_p7 = {"Alec Baldwin": [1,0,0,0,0,0], "Steve Carell": [0,1,0,0,0,0], "Bill Hader": [0,0,1,0,0,0], "Lorraine Bracco": [0,0,0,1,0,0], "Angie Harmon": [0,0,0,0,1,0], "Peri Gilpin": [0,0,0,0,0,1]}
learner.test(testactors_p7, testanswers_p7, thetas_p7)


# Part 8: Plotting Thetas for Part 7
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(thetas_p7[1:, i].reshape((32,32)))
    plt.title(actors_orig[i])
plt.suptitle("Part 7")
plt.show()