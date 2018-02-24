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

encoding = {'bracco':np.array([1,0,0,0,0,0]),
    'gilpin': np.array([0,1,0,0,0,0]),
    'harmon' : np.array([0,0,1,0,0,0]),
    'carell' : np.array([0,0,0,1,0,0]),
    'baldwin' : np.array([0,0,0,0,1,0]),
    'hader' : np.array([0,0,0,0,0,1])}
actors = ['bracco', 'gilpin', 'harmon', 'carell', 'baldwin', 'hader']
ordering = []
num_actors = 6
def loadImages():
    global encoding, actors, ordering, num_actors

    imgs = np.zeros(shape = (227 * 227 * 3,num_actors * 120)) #n-1 = 1024, m = num_actors*120 = 720 training sets
    labels = np.zeros(shape = (num_actors, num_actors * 120)) #k x m
    #theta = np.random.randn(1025 * 6).reshape((1025,6))

    person = 0
    for folder in os.listdir('shrunk/'):
        count = 0
        if folder in actors:
            ordering.append(folder)
            for file in os.listdir('shrunk/'+folder):
                if count < 120:
                    try:
                        imgs[:, count + 120 * person] = imresize(imread('shrunk/'+folder+'/'+file), (227,227)).flatten()
                        labels[:, count + 120 * person] = encoding[folder]
                        count += 1
                    except:
                        continue
                else:
                    break
            person += 1

    
    trainingset = np.zeros(shape = (227 * 227 * 3, num_actors * 80))
    traininglabels = np.zeros(shape = (num_actors, num_actors * 80))

    validationset = np.zeros(shape = (227 * 227 * 3, num_actors*20))
    validationlabels = np.zeros(shape = (num_actors, num_actors*20))

    testset = np.zeros(shape = (227 * 227 * 3, num_actors*20))
    testlabels = np.zeros(shape = (num_actors, num_actors * 20))

    for i in range(num_actors):
        trainingset[:, 80*i:80*(i+1)] = imgs[:, 120*i:120*i + 80]
        traininglabels[:, 80*i:80*(i+1)] = labels[:, 120*i:120*i + 80]

        validationset[:, 20*i:20*(i+1)] = imgs[:, 120*i + 80: 120*i + 80 + 20]
        validationlabels[:, 20*i:20*(i+1)] = labels[:, 120*i + 80:120*i + 80 + 20]

        testset[:, 20*i:20*(i+1)] = imgs[:, 120*i + 80 + 20: 120*i + 80 + 20 + 20]
        testlabels[:, 20*i:20*(i+1)] = labels[:, 120*i + 80 + 20:120*i + 80 + 20 + 20]

    return trainingset.T, traininglabels.T, validationset.T, validationlabels.T, testset.T, testlabels.T


class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8]# MODIFIED WEIGHT LOADING
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

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
            nn.Conv2d(384, 256, kernel_size=3, padding=1)#CONV4
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x) #RETURN JUST FEATURES OF CONV4
        return x

def part10():
    print "----------------------STARTING PART 10------------------------"
    np.random.seed(1)
    torch.manual_seed(0)

    trainingset, traininglabels, validationset, validationlabels, testset, testlabels = loadImages()
    
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

    validationsaving = np.zeros(shape = (120, 43264))
    for index, unprocessed in enumerate(validationset):
        im = unprocessed.reshape(227,227,3) - np.mean(unprocessed.flatten())
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False) 
        validationsaving[index, :] = model.forward(im_v).data.numpy().flatten()

    print 'done converting validation set'

    dim_x = 43264
    dim_out = 6
    dim_h = 64


    print "----------------STARTING NEW NETWORK---------------------"

    learning_rate = 1e-4
    loss_fn = torch.nn.CrossEntropyLoss()

    testvar = Variable(torch.from_numpy(testsaving), requires_grad=False).type(dtype_float)
    validationvar = Variable(torch.from_numpy(validationsaving), requires_grad=False).type(dtype_float)
    trainingvar = Variable(torch.from_numpy(saving), requires_grad=False).type(dtype_float)

    setdivisions = 15
    Xlearn = np.zeros(shape = (setdivisions, 1))
    Y1learn = np.zeros(shape = (setdivisions, 1))
    Y2learn = np.zeros(shape = (setdivisions, 1))
    Y3learn = np.zeros(shape = (setdivisions, 1))

    model2 = None
    for i in range(1, setdivisions+1, 1):
        print "---------iter : " + str(i) + " --------------"
        train_idx = np.random.permutation(range(saving.shape[0]))[:480  * i/setdivisions] #480 for currently using all images
        x = Variable(torch.from_numpy(saving[train_idx]), requires_grad=False).type(dtype_float)
        y_classes = Variable(torch.from_numpy(np.argmax(traininglabels[train_idx], 1)), requires_grad=False).type(dtype_long)

        model2= torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )
        optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
        for t in range(250):
            y_pred = model2(x)
            loss = loss_fn(y_pred, y_classes)
            
            model2.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to step
            
        Xlearn[i-1] = 480  * i/setdivisions
        y_pred = model2(validationvar).data.numpy()
        Y2learn[i-1] = np.mean(np.argmax(y_pred, 1) == np.argmax(validationlabels, 1))
        y_pred = model2(trainingvar).data.numpy()
        Y1learn[i-1] = np.mean(np.argmax(y_pred, 1) == np.argmax(traininglabels, 1))
        
        print "examples: " +str(Xlearn[i-1])
        print "validation set: " + str(Y2learn[i-1])
        print "full training set: " + str(Y1learn[i-1])

        #FOR GENERATING ACCURACY w.r.t ITERATIONS
        #     if t % 40 == 0:
        #         X[t / 40] = t
        #         y_pred = model2(testvar).data.numpy()
        #         Y1[t/40] =  np.mean(np.argmax(y_pred, 1) == np.argmax(testlabels, 1))

        #         y_pred = model2(validationvar).data.numpy()
        #         Y2[t/40] = np.mean(np.argmax(y_pred, 1) == np.argmax(validationlabels, 1))

        #         y_pred = model2(trainingvar).data.numpy()
        #         Y3[t/40] = np.mean(np.argmax(y_pred, 1) == np.argmax(traininglabels, 1))
        # plt.plot(X, Y2,'y', label="Validation Set") 
        # plt.plot(X, Y1,'g', label="Test Set") 
        # plt.plot(X, Y3,'r', label="Training Set")
        # plt.xlabel('iterations')
        # plt.ylabel('accuracy')
        # plt.legend(loc='upper left')
        # fig = plt.gcf()
        # fig.savefig('part8fig.png')
        # plt.show()


    plt.plot(Xlearn, Y1learn, 'y', label="training set")
    plt.plot(Xlearn, Y2learn, "g", label ="validation set")
    plt.xlabel("training set size")
    plt.ylabel("accuracy")
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.savefig('part10learningcurve.png')
    plt.show()

    y_pred = model2(testvar).data.numpy()
    print "FINAL TEST SET ACCURACY: " + str(np.mean(np.argmax(y_pred, 1) == np.argmax(testlabels, 1)))


    