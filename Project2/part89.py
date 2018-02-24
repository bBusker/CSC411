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
        batch_xs = np.zeros((0, 28*28*3))
        batch_y_s = np.zeros( (0, 10))
        
        test_k =  ["test"+str(i) for i in range(10)]
        for k in range(10):
            batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:])/255.)  ))
            one_hot = np.zeros(10)
            one_hot[k] = 1
            batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[test_k[k]]), 1))   ))
        return batch_xs, batch_y_s


    def get_train(M):
        batch_xs = np.zeros((0, 28*28*3))
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
    dtype_long = torch.LongTensor\

    train_idx = np.random.permutation(range(trainingset.shape[0]))[:480]
    x = Variable(torch.from_numpy(trainingset[train_idx]), requires_grad=False).type(dtype_float)
    y = Variable(torch.from_numpy(traininglabels), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(traininglabels[train_idx], 1)), requires_grad=False).type(dtype_long)

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X = np.zeros(shape=(100,1))
    Y1 = np.zeros(shape=(100,1))
    Y2 = np.zeros(shape=(100,1))
    Y3 = np.zeros(shape=(100,1))
    for t in range(4000):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to step
        
        # if t % 40 == 0:
        #     X[t / 40] = t
        #     a = Variable(torch.from_numpy(testset), requires_grad=False).type(dtype_float)
        #     y_pred = model(a).data.numpy()
        #     Y1[t/40] =  np.mean(np.argmax(y_pred, 1) == np.argmax(testlabels, 1))

        #     a = Variable(torch.from_numpy(validationset), requires_grad=False).type(dtype_float)
        #     y_pred = model(a).data.numpy()
        #     Y2[t/40] = np.mean(np.argmax(y_pred, 1) == np.argmax(validationlabels, 1))

        #     a = Variable(torch.from_numpy(trainingset), requires_grad=False).type(dtype_float)
        #     y_pred = model(a).data.numpy()
        #     Y3[t/40] = np.mean(np.argmax(y_pred, 1) == np.argmax(traininglabels, 1))


    # plt.plot(X, Y2,'y', label="Validation Set") 
    # plt.plot(X, Y1,'g', label="Test Set") 
    # plt.plot(X, Y3,'r', label="Training Set")
    # plt.xlabel('iterations')
    # plt.ylabel('accuracy')
    # plt.legend(loc='upper left')
    # fig = plt.gcf()
    # fig.savefig('part8fig.png')
    # plt.show()


    act1 = np.sum(np.matmul(model[0].weight.data.numpy().T, model[2].weight.data.numpy().T[:,3]).reshape(32,32,3),2)
    plt.imshow(act1, cmap=plt.cm.coolwarm)
    fig = plt.gcf()
    fig.savefig('part9fig1.png')
    plt.show()

    act1 = np.sum(np.matmul(model[0].weight.data.numpy().T, model[2].weight.data.numpy().T[:,4]).reshape(32,32,3),2)
    plt.imshow(act1, cmap=plt.cm.coolwarm)
    fig = plt.gcf()
    fig.savefig('part9fig2.png')
    plt.show()