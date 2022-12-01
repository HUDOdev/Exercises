import torch
import torch.nn as nn


class Dropout(nn.Module):
    
    def __init__(self, p=0.1):
        super().__init__()
        # store p
        
    def forward(self, x):
        # In training mode, set each value 
        # independently to 0 with probability p
        # and scale the remaining values 
        # according to the lecture
        # In evaluation mode, return the
        # unmodified input
        pass
    