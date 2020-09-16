from layers import Resnet, Flatten
import torch
import torch.nn as nn
class CNN_MODEL(nn.Module):
    def __init__(self, input_dim=3 , num_filters=16, n_blocks=6):
        super().__init__()
        
        self.models = nn.ModuleList([nn.ReflectionPad2d(4),
                                     nn.Conv2d(input_dim, num_filters, kernel_size=9, stride=1, padding=0)])
        for i in range(n_blocks):
            self.models.append(Resnet(num_filters))
            
        self.models.extend([nn.ReflectionPad2d(1),
                            nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=0),
                            nn.MaxPool2d(4),
                            Flatten(),
                            nn.Linear(16*16, 128),
                            nn.Linear(128, 32),
                            nn.Linear(32, 2),
                            nn.Softmax(1)])
            
    def forward(self, x):
        for layer in self.models:
            x = layer(x)
        return x
