import data_processor
import classifier
import numpy as np
import model

import torch.nn as nn
from torch.autograd import Variable
from torchtext import data
import torch

fake, real = data_processor.loadHeadlines()
train, val, test, embedding, vocab = classifier.prep_data(fake, real)

torch_variables, labels = data_processor.convertTorchVar(fake,real, vocab, 20)

print(torch_variables)
print(len(labels))

# model = model.CNN_Text(embedding, 1, 20, 3)

# print(model.forward(test_var))
