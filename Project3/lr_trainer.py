import random
from math import *
import util 

from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class sets:
    training = 0
    validation = 1
    test = 2

np.random.seed(1)
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

loss_fn = torch.nn.BCELoss()

torch_trainvars = Variable(torch.from_numpy(np_trainingset), requires_grad=False).type(torch.FloatTensor)
torch_trainlabels = Variable(torch.from_numpy(np_traininglabels), requires_grad=False).type(torch.FloatTensor)

model = torch.nn.Sequential(
    torch.nn.Linear(len(all_words), 1),
    torch.nn.Sigmoid()
)

divisions = 40
iterations = 10000

learningCurve = np.zeros(shape = (divisions, 3))
xdim = np.zeros(shape = (divisions))


learning_rate = 8e-4
reg_lambda = 0.021
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_lambda)
print("---------- training linear regression model with Adam ----------")
for t in range(iterations):
    l2_reg = Variable( torch.FloatTensor(1), requires_grad=True)
    for weight in model[0].weight:
        l2_reg = l2_reg + weight.norm(2)

    prediction = model(torch_trainvars)
    # print prediction
    loss = loss_fn(prediction, torch_trainlabels) + reg_lambda * l2_reg
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step
    if t % (iterations / divisions) == 0:
        print "Iteration: " + str(t) + " Loss: " + str(loss)

        xdim[t/(iterations/divisions)] = t

        y_pred = model(Variable(torch.from_numpy(np_trainingset), requires_grad=False).type(torch.FloatTensor)).data.numpy()
        learningCurve[t/(iterations/divisions), 0] = np.mean(np.round(y_pred, 0).flatten() == np_traininglabels)     
        # print np.round(y_pred, 0).flatten()
        # print np_traininglabels 

        y_pred = model(Variable(torch.from_numpy(np_validationset), requires_grad=False).type(torch.FloatTensor)).data.numpy()
        learningCurve[t/(iterations/divisions), 1] = np.mean(np.round(y_pred, 0).flatten() == np_validationlabels)

        y_pred = model(Variable(torch.from_numpy(np_testset), requires_grad=False).type(torch.FloatTensor)).data.numpy()
        learningCurve[t/(iterations/divisions), 2] =np.mean(np.round(y_pred, 0).flatten() == np_testlabels)

plt.plot(xdim, learningCurve[:,0], 'b', label="Training Curve")
plt.plot(xdim, learningCurve[:,1], 'r', label="Validation Set")
plt.plot(xdim, learningCurve[:,2], 'g', label="Test Set")
plt.xlabel("Iterations")
plt.xlabel("Accuracy (%)")
plt.legend(loc='best')
fig = plt.gcf()
fig.savefig('part4learningcurve.png')
# plt.show()

#----------extracting thetas for part6------------

weights = model[0].weight.data.numpy()[0]

real_indices = []
fake_indices = []

for i in range(100):
    fake_indices.append(np.argmax(weights))
    weights[np.argmax(weights)]= 0

weights = model[0].weight.data.numpy()[0]
for i in range(100):
    real_indices.append(np.argmin(weights))
    weights[np.argmin(weights)] = 0

print "----------EXTRACTED WORDS----------"
print "----------fakewords------------"

count = 0
# print all_words.keys()
for index in fake_indices:
    if count == 10:
        break

    print word_order[index]
    count += 1

print "----------realwords----------"

count = 0
for index in real_indices:
    if count == 10:
        break

    print word_order[index]
    count += 1

#----------extracting thetas for part6 NO STOPWORDS------------
print "---------NO STOPWORDS----------"
print "----------fakewords------------"

count = 0
# print all_words.keys()
for index in fake_indices:
    if count == 10:
        break
    
    if word_order[index] in util.ENGLISH_STOP_WORDS:
        continue

    print word_order[index]
    count += 1

print "----------realwords----------"

count = 0
for index in real_indices:
    if count == 10:
        break
    
    if word_order[index] in util.ENGLISH_STOP_WORDS:
        continue

    print word_order[index]
    count += 1
 

