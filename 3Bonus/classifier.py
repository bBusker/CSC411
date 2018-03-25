import random
from math import *
import os

from torch.autograd import Variable
from torchtext import data
import torch
import numpy as np
import matplotlib.pyplot as plt
import data_processor
import model

def train(model, torch_trainvars, torch_trainlabels):
    iterations = 10000

    loss_fn = torch.nn.BCELoss()
    learning_rate = 8e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_lambda)

    for t in range(iterations):
        prediction = model(torch_trainvars)
        loss = loss_fn(prediction, torch_trainlabels)

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to
                            # make a step
    return model

# def testNN(model, test_variables, test_labels):
#     prediction = model(test_variables).data.numpy
#     return np.mean(np.round(y_pred, 0).flatten() == np_validationlabels)


def prep_data(fake_headlines, real_headlines):

    split_ratio = 0.15

    sentence = data.Field(
        sequential=True,
        fix_length=20,
        tokenize=data_processor.clean,
        pad_first=True,
        tensor_type=torch.LongTensor,
        lower=True
    )

    label = data.Field(
        sequential=False,
        use_vocab=False,
        tensor_type=torch.ByteTensor
    )

    fields = [('sentence_text', sentence), ('label', label)]

    examples = []

    headlines = fake_headlines + real_headlines
    labels = [0]*len(fake_headlines) + [1]*len(real_headlines)

    for item in zip(headlines, labels):
        example = data.Example.fromlist(item, fields)
        examples.append(example)

    #random.shuffle(examples)

    sentence.build_vocab(data.Dataset(examples, fields),
                         min_freq=3,
                         vectors="glove.6B.50d"
    )

    vocab = sentence.vocab

    embedding = torch.nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=50, #TODO: change depending on final used word2vec
    )
    embedding.weight.data.copy_(vocab.vectors)

    temp = list(zip(headlines, labels))
    random.shuffle(temp)
    headlines, labels = zip(*temp)

    test_split = int(len(headlines)*split_ratio)
    val_split = int(len(headlines)*split_ratio)+test_split

    train = headlines[val_split:]
    val = headlines[test_split:val_split]
    test = headlines[:test_split]

    train_labels = labels[val_split:]
    val_labels = labels[test_split:val_split]
    test_labels = labels[:test_split]

    train = sentence.process(train, -1, True)
    val = sentence.process(val, -1, False)
    test = sentence.process(test, -1, False)

    return train, val, test, train_labels, val_labels, test_labels, embedding, vocab.stoi