import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):
    
    def __init__(self, embedding, output_channels, embedding_length, kernel_width):
        super(CNN_Text, self).__init__()

        self.embedding = embedding
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(100 ,1), stride=1, padding=(0, 0))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(100, 2), stride=1, padding=(0, 0))
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(100, 3), stride=1, padding=(0, 0))
        self.conv4 = nn.Conv2d(1, 1, kernel_size=(100, 4), stride=1, padding=(0, 0))
        self.conv5 = nn.Conv2d(1, 1, kernel_size=(100, 5), stride=1, padding=(0, 0))
        self.max1 = nn.MaxPool1d(17)
        self.max2 = nn.MaxPool1d(16)
        self.max3 = nn.MaxPool1d(15)
        self.max4 = nn.MaxPool1d(14)
        self.max5 = nn.MaxPool1d(13)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = x.permute(0,1,3,2)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        x3 = x3.squeeze(1)
        x4 = x4.squeeze(1)
        x5 = x5.squeeze(1)
        x1 = self.max1(x1)
        x2 = self.max2(x2)
        x3 = self.max3(x3)
        x4 = self.max4(x4)
        x5 = self.max5(x5)
        x = torch.cat((x1,x2,x3,x4,x5), 2)
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = x.squeeze(1)
        return x