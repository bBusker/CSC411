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

# def train():
#     iterations = 10000
#
#     loss_fn = torch.nn.BCELoss()
#     learning_rate = 8e-2
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_lambda)
#
#     model = model.CNN_Text()
#
#     for t in range(iterations):
#         prediction = model(torch_trainvars)
#         loss = loss_fn(prediction, torch_trainlabels)
#
#         model.zero_grad()  # Zero out the previous gradient computation
#         loss.backward()    # Compute the gradient
#         optimizer.step()   # Use the gradient information to
#                             # make a step
#
# def testNN(model, test_variables, test_labels)
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

    random.shuffle(examples)

    test_split = int(len(examples)*split_ratio)
    val_split = int(len(examples)*split_ratio)+test_split

    train = data.Dataset(examples[val_split:], fields)
    val = data.Dataset(examples[test_split:val_split], fields)
    test = data.Dataset(examples[:test_split], fields)

    sentence.build_vocab(train,
                         min_freq=3,
                         vectors="glove.6B.50d"
    )
    sentence.build_vocab()

    vocab = sentence.vocab

    embedding = torch.nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=50, #TODO: change depending on final used word2vec
    )
    embedding.weight.data.copy_(vocab.vectors)

    return train, val, test, embedding, vocab.stoi

