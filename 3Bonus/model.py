import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):
    
    def __init__(self, embedding, output_channels, embedding_length, kernel_width):
        super(CNN_Text, self).__init__()

        self.embedding = embedding
        self.conv1 = nn.Conv2d(1, output_channels, kernel_size=(embedding_length,kernel_width), stride=1, padding=(0,2))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.linear1 = nn.Linear(52, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.squeeze(2)
        x = x.squeeze(1)
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x