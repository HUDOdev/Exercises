import torch
import torch.nn as nn


class Dropout(nn.Module):
    
    def __init__(self, p=0.1):
        super().__init__()
        # store p
        self.p = p
        
    def forward(self, x):
        # In training mode, set each value 
        # independently to 0 with probability p
        # and scale the remaining values 
        # according to the lecture
        # In evaluation mode, return the
        # unmodified input

        if self.training:
            mask = torch.ones_like(x)
            noise = torch.rand_like(x)
            mask[noise <=self.p] = 0
            mask = mask / (1-self.p)
            return x * mask
        else:
            return x
