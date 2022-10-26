import numpy as np


class Module:
    
    def forward(self, *args, **kwargs):
        pass

    
class Network(Module):
    
    def __init__(self, layers=None):
        # store the list of layers passed in the constructor in your Network object
        pass
    
    def forward(self, x):
        # for executing the forward pass, run the forward passes of each
        # layer and pass the output as input to the next layer
        pass
    
    def add_layer(self, layer):
        # append layer at the end of the list of already existing layer
        pass

    
class LinearLayer(Module):
    
    def __init__(self, W, b):
        # store parameters W and b
        self.W = W
        self.b = b
        pass
    
    def forward(self, x):
        # compute the affine linear transformation x -> Wx + b
        return self.W @ x + self.b
        pass

    
class Sigmoid(Module):
    
    def forward(self, x):
        # implement the sigmoid
        self.x = np.exp(x)/(1+np.exp(x))
        return self.x
        pass

    
class ReLU(Module):
    
    def forward(self, x):
        # implement a ReLU
        pass

    
class Loss(Module):
    
    def forward(self, prediction, target):
        pass


class MSE(Loss):
    
    def forward(self, prediction, target):
        # implement MSE loss
        pass


class CrossEntropyLoss(Loss):
    
    def forward(self, prediction, target):
        # implement cross entropy loss
        pass
