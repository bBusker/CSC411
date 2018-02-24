from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy import float32
from scipy.misc import imread
from matplotlib.pyplot import imsave
from scipy.misc import imresize

import part89 
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8]#, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        # classifier_weight_i = [1, 4, 6]
        # for i in classifier_weight_i:
        #     self.classifier[i].weight = an_builtin.classifier[i].weight
        #     self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        return x

def part10():
    print "----------------------STARTING PART 10------------------------"
    trainingset, traininglabels, validationset, validationlabels, testset, testlabels = part89.loadImages(227,227)
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    model = MyAlexNet()
    model.eval()

    softmax = torch.nn.Softmax()

    saving = np.zeros(shape = (480, 43264))
    # Read an image
    for index, unprocessed in enumerate(trainingset):
        im = unprocessed.reshape(227,227,3) - np.mean(unprocessed.flatten())
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False) 
        saving[index, :] = model.forward(im_v).data.numpy().flatten()

    print('done converting trainingset')

    testsaving = np.zeros(shape = (120, 43264))
    for index, unprocessed in enumerate(testset):
        im = unprocessed.reshape(227,227,3) - np.mean(unprocessed.flatten())
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False) 
        testsaving[index, :] = model.forward(im_v).data.numpy().flatten()

    print 'done converting testset'
    print testsaving.shape
    print saving.shape

    validationsaving = np.zeros(shape = (120, 43264))
    for index, unprocessed in enumerate(validationset):
        im = unprocessed.reshape(227,227,3) - np.mean(unprocessed.flatten())
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32).flatten()
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False) 
        validationsaving[index, :] = model.forward(im_v).data.numpy().flatten()


    dim_x = 43264
    dim_out = 6
    dim_h = 32


    print "----------------STARTING NEW NETWORK---------------------"
    train_idx = np.random.permutation(range(trainingset.shape[0]))[:480] #480 for currently using all images
    x = Variable(torch.from_numpy(saving[train_idx]), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(traininglabels[train_idx], 1)), requires_grad=False).type(dtype_long)

    model2 = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    X = np.zeros(shape=(100,1))
    Y1 = np.zeros(shape=(100,1))
    Y2 = np.zeros(shape=(100,1))
    Y3 = np.zeros(shape=(100,1))
    max_iter = 400
    for t in range(max_iter):
        print "iter:" + str(t)
        y_pred = model2(x)
        loss = loss_fn(y_pred, y_classes)
        
        model2.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to step
        
        if t % (max_iter / 100) == 0:
            X[t / (max_iter/100)] = t
            a = Variable(torch.from_numpy(testsaving), requires_grad=False).type(dtype_float)
            y_pred = model2(a).data.numpy()
            Y1[t/(max_iter/100)] =  np.mean(np.argmax(y_pred, 1) == np.argmax(testlabels, 1))

            a = Variable(torch.from_numpy(validationsaving), requires_grad=False).type(dtype_float)
            y_pred = model2(a).data.numpy()
            Y2[t/(max_iter/100)] = np.mean(np.argmax(y_pred, 1) == np.argmax(validationlabels, 1))

            a = Variable(torch.from_numpy(saving), requires_grad=False).type(dtype_float)
            y_pred = model2(a).data.numpy()
            Y3[t/(max_iter/100)] = np.mean(np.argmax(y_pred, 1) == np.argmax(traininglabels, 1))

    plt.plot(X, Y2,'y', label="Validation Set") 
    plt.plot(X, Y1,'g', label="Test Set") 
    plt.plot(X, Y3,'r', label="Training Set")
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.savefig('part10fig.png')
    plt.show()


    