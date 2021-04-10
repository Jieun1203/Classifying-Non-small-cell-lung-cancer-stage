import os
import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, flatten_size, out_features1,
                 out_features2, out_features3, out_features4, out_features5):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels1, kernel_size= 3, padding = 1)
        self.cnn2 = nn.Conv2d(out_channels1, out_channels2, kernel_size= 3, padding = 1)
        self.linear1 = nn.Linear(flatten_size, out_features1)
        self.linear2 = nn.Linear(out_features1, out_features2)
        self.linear3 = nn.Linear(out_features2, out_features3)
        self.linear4 = nn.Linear(out_features3, out_features4)
        self.linear5 = nn.Linear(out_features4, out_features5)
        self.BatchNorm1 = nn.BatchNorm2d(out_channels1)
        self.BatchNorm2 = nn.BatchNorm2d(out_channels2)
        self.dropout = nn.Dropout2d(0.5)
        self.dropout1 = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.average_pool = nn.AvgPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        output = self.cnn1(x)
        output = self.BatchNorm1(output)
        output = self.relu(output)
        output = self.average_pool(output)
        output = self.cnn2(output)
        output = self.BatchNorm2(output)
        output = self.relu(output)
        output = self.average_pool(output)
        output = output.view(output.size(0), -1)
        output = self.linear1(output)
        output = self.dropout1(output)
        output = self.linear2(output)
        output = self.dropout1(output)
        output = self.linear3(output)
        output = self.dropout1(output)
        output = self.linear4(output)
        output = self.dropout1(output)
        output = self.linear5(output)
        output = F.softmax(output, dim = 1)
        return output
