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