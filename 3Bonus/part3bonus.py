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

embedding_length = 17

# fake, real = data_processor.loadHeadlines()
f_real = open("Data/clean_real.txt")
f_fake = open("Data/clean_fake.txt")

real = [str.split(line) for line in f_real]
fake = [str.split(line) for line in f_fake]


train, val, test, train_labels, val_labels, test_labels, embedding, vocab = classifier.prep_data(fake, real, embedding_length)

train = train.transpose(0,1)
test = test.transpose(0,1)
val = val.transpose(0,1)
train_labels = Variable(torch.FloatTensor(train_labels))

model = model.CNN_Text(embedding, 1, embedding_length, 3)
#model = model.CNN_Text2(embedding)
model = classifier.train(model, train, train_labels, val, val_labels)

model.training = False

torch.save(model.state_dict(), 'model_state_dict')

print("test set acc: {}".format(classifier.testNN(model, test, test_labels)))