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

def loadImages():
    global encoding, actors, ordering, num_actors

    imgs = np.zeros(shape = (32 * 32 * 3,num_actors * 120)) #n-1 = 1024, m = num_actors*120 = 720 training sets
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
                        imgs[:, count + 120 * person] = imresize(imread('shrunk/'+folder+'/'+file), (32,32)).flatten()
                        labels[:, count + 120 * person] = encoding[folder]
                        count += 1
                    except:
                        continue
                else:
                    break
            person += 1

    imgs = imgs/255.0
    
    trainingset = np.zeros(shape = (3072, num_actors * 80))
    traininglabels = np.zeros(shape = (num_actors, num_actors * 80))

    validationset = np.zeros(shape = (3072, num_actors*20))
    validationlabels = np.zeros(shape = (num_actors, num_actors*20))

    testset = np.zeros(shape = (3072, num_actors*20))
    testlabels = np.zeros(shape = (num_actors, num_actors * 20))

    for i in range(num_actors):
        trainingset[:, 80*i:80*(i+1)] = imgs[:, 120*i:120*i + 80]
        traininglabels[:, 80*i:80*(i+1)] = labels[:, 120*i:120*i + 80]

        validationset[:, 20*i:20*(i+1)] = imgs[:, 120*i + 80: 120*i + 80 + 20]
        validationlabels[:, 20*i:20*(i+1)] = labels[:, 120*i + 80:120*i + 80 + 20]

        testset[:, 20*i:20*(i+1)] = imgs[:, 120*i + 80 + 20: 120*i + 80 + 20 + 20]
        testlabels[:, 20*i:20*(i+1)] = labels[:, 120*i + 80 + 20:120*i + 80 + 20 + 20]

    return trainingset.T, traininglabels.T, validationset.T, validationlabels.T, testset.T, testlabels.T

def part89():
    trainingset, traininglabels, validationset, validationlabels, testset, testlabels = loadImages()

    np.random.seed(1)
    torch.manual_seed(0)

    dim_x = 32 * 32 * 3
    dim_out = 6
    dim_h = 24
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    

    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-4
    
    #for generating accuracy vs sample size w/ constant iterations
    
    testvar = Variable(torch.from_numpy(testset), requires_grad=False).type(dtype_float)
    validationvar = Variable(torch.from_numpy(validationset), requires_grad=False).type(dtype_float)
    trainingvar = Variable(torch.from_numpy(trainingset), requires_grad=False).type(dtype_float)
    
    setdivisions = 3
    Xlearn = np.zeros(shape = (setdivisions, 1))
    Y1learn = np.zeros(shape = (setdivisions, 1))
    Y2learn = np.zeros(shape = (setdivisions, 1))
    Y3learn = np.zeros(shape = (setdivisions, 1))

    model = None
    for i in range(1, setdivisions+1, 1):
        train_idx = np.random.permutation(range(trainingset.shape[0]))[:480  * i/setdivisions] #480 for currently using all images
        x = Variable(torch.from_numpy(trainingset[train_idx]), requires_grad=False).type(dtype_float)
        y_classes = Variable(torch.from_numpy(np.argmax(traininglabels[train_idx], 1)), requires_grad=False).type(dtype_long)

        model = torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for t in range(3000):
            y_pred = model(x)
            loss = loss_fn(y_pred, y_classes)
            
            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to step
            
        Xlearn[i-1] = 480  * i/setdivisions
        y_pred = model(validationvar).data.numpy()
        Y2learn[i-1] = np.mean(np.argmax(y_pred, 1) == np.argmax(validationlabels, 1))
        y_pred = model(trainingvar).data.numpy()
        Y1learn[i-1] = np.mean(np.argmax(y_pred, 1) == np.argmax(traininglabels, 1))
        print "---------iter : " + str(i) + " --------------"
        print "examples: " +str(Xlearn[i-1])
        print "validation set: " + str(Y2learn[i-1])
        print "full training set: " + str(Y1learn[i-1])

    y_pred = model(testvar).data.numpy()
    plt.plot(Xlearn, Y1learn, 'y', label="training set")
    plt.plot(Xlearn, Y2learn, "g", label ="validation set")
    plt.xlabel("training set size")
    plt.ylabel("accuracy")
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.savefig('part8learningcurvesamplesize.png')
    plt.show()
    print "FINAL TEST SET ACCURACY: " + str(np.mean(np.argmax(y_pred, 1) == np.argmax(testlabels, 1)))


    matrix = np.zeros(shape = (dim_h, 1))
    matrix[np.argmax(model[2].weight.data.numpy().T[:,5])] = 1
    act1 = np.sum(np.matmul(model[0].weight.data.numpy().T, matrix).reshape(32,32,3),2)
    plt.imshow(act1, cmap=plt.cm.coolwarm)
    fig = plt.gcf()
    fig.savefig('part9fig1.png')
    plt.show()

    matrix = np.zeros(shape = (dim_h, 1))
    matrix[np.argmax(model[2].weight.data.numpy().T[:,4])] = 1
    act1 = np.sum(np.matmul(model[0].weight.data.numpy().T, matrix).reshape(32,32,3),2)
    plt.imshow(act1, cmap=plt.cm.coolwarm)
    fig = plt.gcf()
    fig.savefig('part9fig2.png')
    plt.show()
    
    #generate accuracy vs iterations
    y_classes = Variable(torch.from_numpy(np.argmax(traininglabels, 1)), requires_grad=False).type(dtype_long)

    X = np.zeros(shape=(100,1))
    Y1 = np.zeros(shape=(100,1))
    Y2 = np.zeros(shape=(100,1))
    Y3 = np.zeros(shape=(100,1))

    model = torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(4000):
            y_pred = model(trainingvar)
            loss = loss_fn(y_pred, y_classes)
            
            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to step

            # FOR GENERATING ACCURACY w.r.t ITERATIONS
            if t % 40 == 0:
                X[t / 40] = t
                y_pred = model(testvar).data.numpy()
                Y1[t/40] =  np.mean(np.argmax(y_pred, 1) == np.argmax(testlabels, 1))

                y_pred = model(validationvar).data.numpy()
                Y2[t/40] = np.mean(np.argmax(y_pred, 1) == np.argmax(validationlabels, 1))

                y_pred = model(trainingvar).data.numpy()
                Y3[t/40] = np.mean(np.argmax(y_pred, 1) == np.argmax(traininglabels, 1))

    plt.plot(X, Y2,'y', label="Validation Set") 
    plt.plot(X, Y1,'g', label="Test Set") 
    plt.plot(X, Y3,'r', label="Training Set")
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.savefig('part8iterationfig.png')
    plt.show()