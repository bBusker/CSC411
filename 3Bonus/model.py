import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):
    
    def __init__(self, example_length, embedding, kernel_width, output_channels):
        super(CNN_Text, self).__init__()

        self.embedding_length = embedding_length
        self.word_length = word_length
        self.output_channels = output_channels

        self.layers = nn.Sequential(
            nn.Conv2d(1, output_channels, kernel_size=(embedding_length, kernel_width), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(output_channels * example_length - kernel_width + 1, 1),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x