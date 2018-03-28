import random
from math import *
import os
import pickle

from torch.autograd import Variable
from torchtext import data
import torch
import numpy as np
import matplotlib.pyplot as plt
import data_processor
import model

def train(model, train_vars, train_labels, val_vars, test_labels):
    iterations = 400

    model.training=True

    loss_fn = torch.nn.BCELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00012)

    for t in range(iterations):
        prediction = model(train_vars)
        loss = loss_fn(prediction, train_labels)

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to
                            # make a step
        if t % 10 == 0:
            print("iter: {} --------".format(t))
            print("  loss: {:.4f}".format(loss.data[0]))
            print("  acc: {:.4f}".format(testNN(model, val_vars, test_labels)))

    return model

def testNN(model, test_variables, test_labels):
    prev = model.training
    model.training=False
    prediction = model(test_variables).data.numpy()
    model.training=prev
    return np.mean(np.round(prediction, 0).flatten() == np.asarray(test_labels))


def prep_data(fake_headlines, real_headlines, embedding_length):
    random.seed(0)
    split_ratio = 0.15

    sentence = data.Field(
        sequential=True,
        fix_length=embedding_length,
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

    # random.shuffle(examples)

    sentence.build_vocab(data.Dataset(examples, fields),
                         min_freq=3,
                         vectors="glove.6B.100d"
    )

    vocab = sentence.vocab

    embedding = torch.nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=100, #TODO: change depending on final used word2vec
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

    with open('embedding_layer.pkl', 'wb') as f:
        pickle.dump(embedding, f, pickle.HIGHEST_PROTOCOL)

    with open('vocabstoi.pkl', 'wb') as f:
        pickle.dump(vocab.stoi, f, pickle.HIGHEST_PROTOCOL)

    return train, val, test, train_labels, val_labels, test_labels, embedding, vocab.stoi