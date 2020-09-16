import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Resnet(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        
        self.models = nn.Sequential(nn.ReflectionPad2d(2),
                                    nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding=0),
                                    nn.InstanceNorm2d(num_filters),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(3),
                                    nn.Conv2d(num_filters, num_filters, kernel_size=7, stride=1, padding=0),
                                    nn.InstanceNorm2d(num_filters))
        
    def forward(self, x):
        x = x + self.models(x) 
        return x
