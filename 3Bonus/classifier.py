import random
from math import *
import util 

from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def train():
    loss_fn = torch.nn.BCELoss()
    learning_rate = 8e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_lambda)

    model = torch.nn.Sequential(
    torch.nn.Linear(len(all_words), 1),
    torch.nn.Sigmoid()
    )   

    for t in range(iterations):
        prediction = model(torch_trainvars)
        loss = loss_fn(prediction, torch_trainlabels) + reg_lambda * l2_reg

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                            # make a step
# import data_processor
# import numpy
# from torchtext import data
# import torch

# templist = [["sample", "headline", "one"], ["sample", "headline", "two"]]


# def prep_data():
#     sentence = data.Field(
#         sequential=True,
#         fix_length=20,
#         tokenize=data_processor.clean,
#         pad_first=True,
#         tensor_type=torch.LongTensor,
#         lower=True
#     )

#     label = data.Field(
#         sequential=False,
#         use_vocab=False,
#         tensor_type=torch.ByteTensor
#     )

#     fields = [('sentence_text', sentence), ('label', label)]

#     headlines = []
#     for temp in templist:
#         headline = data.Example.fromlist((temp, 1), fields)
#         headlines.append(headline)

#     train = data.Dataset(headlines, fields)
#     val = data.Dataset(headlines, fields)
#     test = data.Dataset(headlines, fields)

#     sentence.build_vocab(train, val, test,
#                          max_size=100,
#                          min_freq=1,
#                          vectors="glove.6B.50d"
#     )

#     return train, val, test
