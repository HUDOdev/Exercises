import torch
import torch.nn as nn


# modify the edge detector kernel in such a way that
# it calculates the derivatives in x and y direction
edge_detector_kernel = torch.zeros(2, 1, 2, 2)

edge_detector_kernel[0,0,:,:] = torch.tensor([[-1,0],
                                                [1,0]])
        
edge_detector_kernel[1,0,:,:] = torch.tensor([[-1,1],
                                                [0,0]])


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

        C, H, W = x.shape
        C_out, C_in, h, w = self.kernel.shape
        
        H_out = (H-h) // self.stride + 1
        W_out = (W-w) // self.stride + 1

        x_out = torch.zeros(C_out,H_out,W_out)

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:,i*self.stride:i*self.stride+h,j*self.stride:j*self.stride+w] * self.kernel
                x_out[:,i,j] = patch.sum(dim=([1,2,3]))
        
        return x_out

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

        
        C, H, W = x.shape

        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding

        x_pad = torch.zeros(C,H_pad,W_pad)
        x_pad[:,self.padding:H+self.padding, self.padding:W+self.padding] = x

        return x_pad
        
