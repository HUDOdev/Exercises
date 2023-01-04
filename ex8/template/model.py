import torch
import torch.nn as nn


class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #############################
        # Initialize your network
        #############################
        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        #############################
        
        pass
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model')

