import data_processor
import classifier
import numpy as np
import model

import torch.nn as nn
from torch.autograd import Variable
from torchtext import data
import torch
import random

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

fake, real = data_processor.loadHeadlines()
train, val, test, train_labels, val_labels, test_labels, embedding, vocab = classifier.prep_data(fake, real)

train = train.transpose(0,1)
test = test.transpose(0,1)
val = val.transpose(0,1)
train_labels = Variable(torch.FloatTensor(train_labels))

model = model.CNN_Text(embedding, 1, 20, 3)
model = classifier.train(model, train, train_labels, test, test_labels)

print(classifier.testNN(model, test, test_labels))
