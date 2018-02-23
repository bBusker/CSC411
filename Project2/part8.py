from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.misc import imread
from matplotlib.pyplot import imsave
from scipy.misc import imresize

encoding = {'bracco':np.array([1,0,0,0,0,0]),
    'gilpin': np.array([0,1,0,0,0,0]),
    'harmon' : np.array([0,0,1,0,0,0]),
    'carell' : np.array([0,0,0,1,0,0]),
    'baldwin' : np.array([0,0,0,0,1,0]),
    'hader' : np.array([0,0,0,0,0,1])}
    
actors = ['bracco', 'gilpin', 'harmon', 'carell', 'baldwin', 'hader']
ordering = []
num_actors = 6

def testshit():
    from scipy.io import loadmat
    M = loadmat("mnist_all.mat")
    def get_test(M):
        batch_xs = np.zeros((0, 28*28))
        batch_y_s = np.zeros( (0, 10))
        
        test_k =  ["test"+str(i) for i in range(10)]
        for k in range(10):
            batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:])/255.)  ))
            one_hot = np.zeros(10)
            one_hot[k] = 1
            batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[test_k[k]]), 1))   ))
        return batch_xs, batch_y_s


    def get_train(M):
        batch_xs = np.zeros((0, 28*28))
        batch_y_s = np.zeros( (0, 10))
        
        train_k =  ["train"+str(i) for i in range(10)]
        for k in range(10):
            batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:])/255.)  ))
            one_hot = np.zeros(10)
            one_hot[k] = 1
            batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[train_k[k]]), 1))   ))
        return batch_xs, batch_y_s
            

    train_x, train_y = get_train(M)
    test_x, test_y = get_test(M)
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape

def loadImages():
    global encoding, actors, ordering, num_actors

    imgs = np.zeros(shape = (1024,num_actors * 120)) #n-1 = 1024, m = num_actors*120 = 720 training sets
    labels = np.zeros(shape = (num_actors, num_actors * 120)) #k x m
    theta = np.zeros(shape = (1025, 6)) #n * k
    #theta = np.random.randn(1025 * 6).reshape((1025,6))

    person = 0
    for folder in os.listdir('shrunk/'):
        count = 0
        if folder in actors:
            ordering.append(folder)
            for file in os.listdir('shrunk/'+folder):
                if count < 120:
                    try:
                        imgs[:, count + 120 * person] = imread('shrunk/'+folder+'/'+file).flatten()
                        labels[:, count + 120 * person] = encoding[folder]
                        count += 1
                    except:
                        continue
                else:
                    break
            person += 1
    
    trainingset = np.zeros(shape = (1024, num_actors * 80))
    traininglabels = np.zeros(shape = (num_actors, num_actors * 80))

    testset = np.zeros(shape = (1024, num_actors*20))
    testlabels = np.zeros(shape = (num_actors, num_actors * 20))

    for i in range(num_actors):
        trainingset[:, 80*i:80*(i+1)] = imgs[:, 120*i:120*i + 80]
        traininglabels[:, 80*i:80*(i+1)] = labels[:, 120*i:120*i + 80]

        testset[:, 20*i:20*(i+1)] = imgs[:, 120*i + 80 + 20: 120*i + 80 + 20 + 20]
        testlabels[:, 20*i:20*(i+1)] = labels[:, 120*i + 80 + 20:120*i + 80 + 20 + 20]

    return trainingset.T, traininglabels.T, testset.T, testlabels.T

def part8():
    trainingset, traininglabels, testset, testlabels = loadImages()

    dim_x = 32 * 32
    dim_out = 6
    dim_h = 20
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_idx = np.random.permutation(range(trainingset.shape[0]))[:80]
    x = Variable(torch.from_numpy(trainingset[train_idx]), requires_grad=False).type(dtype_float)
    y = Variable(torch.from_numpy(traininglabels), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(traininglabels[train_idx], 1)), requires_grad=False).type(dtype_long)

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(20000):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to step

    x = Variable(torch.from_numpy(testset), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()
    print np.mean(np.argmax(y_pred, 1) == np.argmax(testlabels, 1))