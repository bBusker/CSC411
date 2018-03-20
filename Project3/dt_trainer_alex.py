import random
from math import *
import util 

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import tree

class sets:
    training = 0
    validation = 1
    test = 2

np.random.seed(0)
torch.manual_seed(0)

fakes, reals = util.generate_sets()

sets_fakes = {sets.training: fakes[:909], sets.validation: fakes[909:1104], sets.test: fakes[1104:1298]}
sets_reals = {sets.training: reals[:1378], sets.validation: reals[1378:1673], sets.test: reals[1673:1968]}

training_examples = sets_fakes[sets.training] + sets_reals[sets.training]
validation_examples = sets_fakes[sets.validation] + sets_reals[sets.validation]
test_examples = sets_fakes[sets.test] + sets_reals[sets.test]

all_words = {} #generate list of all our words present
word_order = {}
for headline in training_examples + validation_examples + test_examples:
    for word in headline:
        if word not in all_words:
            word_order[len(all_words)] = word
            all_words[word] = len(all_words)

np_trainingset = np.zeros(shape=(len(training_examples), len(all_words)))
np_traininglabels = np.zeros(shape=(len(training_examples)))
for index, headline in enumerate(training_examples):
    for word in headline:
        np_trainingset[index, all_words[word]] += 1
    if index < len(sets_fakes[sets.training]):
        np_traininglabels[index] = 0
    else:
        np_traininglabels[index] = 1

np_validationset = np.zeros(shape=(len(validation_examples), len(all_words)))
np_validationlabels = np.zeros(shape=(len(validation_examples)))
for index, headline in enumerate(validation_examples):
    for word in headline:
        np_validationset[index, all_words[word]] += 1
    if index < len(sets_fakes[sets.validation]):
        np_validationlabels[index] = 0
    else:
        np_validationlabels[index] = 1

np_testset = np.zeros(shape=(len(test_examples), len(all_words)))
np_testlabels = np.zeros(shape=(len(test_examples)))
for index, headline in enumerate(test_examples):
    for word in headline:
        np_testset[index, all_words[word]] += 1
    if index < len(sets_fakes[sets.test]):
        np_testlabels[index] = 0
    else:
        np_testlabels[index] = 1


#make decision tree
X = np.zeros(10)
Y = np.zeros(shape = (10, 3))

for index, i in enumerate([2,5,10,20,30,40,50,75,100,200]):
    clf = tree.DecisionTreeClassifier(max_depth = i)
    clf = clf.fit(np_trainingset, np_traininglabels)
    
    X[index] = i
    Y[index, 0] = clf.score(np_trainingset, np_traininglabels)
    Y[index, 1] = clf.score(np_validationset, np_validationlabels)
    Y[index, 2] = clf.score(np_testset, np_testlabels)

plt.plot(X, Y[:,0], 'b', label="Training Curve")
plt.plot(X, Y[:,1], 'r', label="Validation Set")
plt.plot(X, Y[:,2], 'g', label="Test Set")
plt.xlabel("Depth")
plt.ylabel("Accuracy (%)")
plt.legend(loc='best')
fig = plt.gcf()
fig.savefig('part7learningcurve.png')
# plt.show()


#-taking best tree
clf = tree.DecisionTreeClassifier(max_depth = 50)
clf = clf.fit(np_trainingset, np_traininglabels)

dotfile = open("tree.dot", 'w')
tree.export_graphviz(clf, out_file = dotfile, max_depth = 2)
dotfile.close()

for index in [125, 12, 1185, 92, 501, 255]:
    print "index=" + str(index) + " word: " + word_order[index]