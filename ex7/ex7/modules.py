import torch
import torch.nn as nn


# modify the edge detector kernel in such a way that
# it calculates the derivatives in x and y direction
edge_detector_kernel = torch.zeros(2, 1, 2, 2)


class Conv2d(nn.Module):
    
    def __init__(self, kernel, padding=0, stride=1):
        super().__init__()
        self.kernel = nn.Parameter(kernel)
        self.padding = ZeroPad2d(padding)
        self.stride = stride
        
    def forward(self, x):
        x = self.padding(x)
        # For input of shape C x H x W
        # implement the convolution of x with self.kernel
        # using self.stride as stride
        # The output is expected to be of size C x H' x W'
        pass


class ZeroPad2d(nn.Module):
    
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
        
    def forward(self, x):
        # For input of shape C x H x W
        # return tensor zero padded equally at left, right,
        # top, bottom such that the output is of size
        # C x (H + 2 * self.padding) x (W + 2 * self.padding)
        pass
