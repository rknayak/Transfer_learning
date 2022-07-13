
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear, Dropout

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            ,MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            
            ,Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            ,MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            
            ,Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            
            ,Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            ,MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            
            ,Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            
            ,Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            ,MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            
            ,Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            
            ,Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            ,MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.classifier = nn.Sequential(
            Linear(in_features=512*2*2, out_features=256, bias=True)
            ,ReLU(inplace=True)
            ,Dropout(p=0.5, inplace=False)
            
            ,Linear(in_features=256, out_features=128, bias=True)
            ,ReLU(inplace=True)
            ,Dropout(p=0.5, inplace=False)
            
            ,Linear(in_features=128, out_features=10, bias=True)
        )
       
    def forward(self,x):
        
        out = self.features(x)
        out = torch.flatten(out,1)  # we get a tensor with size batch*512*2*2
        out = self.classifier(out)  # size batch*10
        
        return out