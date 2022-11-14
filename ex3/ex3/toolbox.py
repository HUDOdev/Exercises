import numpy as np


class Tensor():
    
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        self.grad_fn = None
        self.num_forward = -1
    
    def backward(self, dLdt=1.):
        if self.requires_grad:
            self.grad += dLdt
            if self.num_forward > 0:
                self.num_forward -= 1
            elif self.grad_fn is not None:
                self.grad_fn.backward(self.grad)
                del self.grad_fn
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def __add__(self, other):
        return Add().forward(self, other)
    
    def __mul__(self, other):
        return Mul().forward(self, other)
    
    def __truediv__(self, other):
        return Div().forward(self, other)
    
    def __pow__(self, exponent):
        return Pow().forward(self, exponent)
    
    def __matmul__(self, other):
        return MatMul().forward(self, other)
    
    def sum(self, dim=0):
        return Sum().forward(self, dim)
    
    def mean(self):
        return Mean().forward(self)
    
    def __getitem__(self, index):
        return GetItem().forward(self, index)


class Function():
    
    def forward(self, *tensors):
        for t in tensors:
            t.num_forward += 1
    
    def backward(self, dLdf):
        pass



class Add(Function):
    
    def forward(self, t1, t2):
        super().forward(t1, t2)
        # storing the references to tensors
        # involved in calculating the result
        self.t1 = t1
        self.t2 = t2
        # creating resulting tensor
        requires_grad = t1.requires_grad or t2.requires_grad
        t = Tensor(t1.data+t2.data, requires_grad=requires_grad)
        # storing reference to the operation
        # creating t, i.e. self
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # applying chain rule to stored tensors
        grad = 1 * dLdf
        self.t1.backward(grad)
        grad = 1 * dLdf
        self.t2.backward(grad)


class Mul(Function):
    
    def forward(self, t1, t2):
        super().forward(t1, t2)
        self.t1 = t1
        self.t2 = t2
        requires_grad = t1.requires_grad or t2.requires_grad
        t = Tensor(t1.data*t2.data, requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of Mul
        grad = 1 * dLdf * self.t2.data
        self.t1.backward(grad)
        grad = 1 * dLdf * self.t1.data
        self.t2.backward(grad)
        pass


class Div(Function):
    
    def forward(self, t1, t2):
        super().forward(t1, t2)
        self.t1 = t1
        self.t2 = t2
        requires_grad = t1.requires_grad or t2.requires_grad
        t = Tensor(t1.data/t2.data, requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of Div
        self.t1.backward(dLdf / self.t2.data)
        self.t2.backward(-1 * dLdf * self.t1.data / self.t2.data **2 )
        pass

            
class Pow(Function):
    
    def forward(self, t, exponent):
        super().forward(t)
        self.t = t
        self.exponent = exponent
        requires_grad = t.requires_grad
        t = Tensor(t.data ** exponent, requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of Pow
        # only consider self.t but not the exponent
        self.t.backward(dLdf * self.exponent * self.t.data ** (self.exponent - 1))
        pass
            

class MatMul(Function):
    
    def forward(self, M, v):
        super().forward(M, v)
        self.M = M
        self.v = v
        requires_grad = M.requires_grad or v.requires_grad
        t = Tensor(M.data@v.data, requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of MatMul
        pass


class ReLU(Function):
    
    def forward(self, t):
        super().forward(t)
        self.t = t
        requires_grad = t.requires_grad
        t = Tensor(np.maximum(t.data, 0), requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of ReLU
        t = self.t.data
        t[t<0] = 0
        t[t>0] = 1
        self.t.backward(dLdf * t)
        pass

def relu(t):
    return ReLU().forward(t)


class Exp(Function):
    
    def forward(self, t):
        super().forward(t)
        self.t = t
        requires_grad = t.requires_grad
        self.expt = np.exp(t.data)
        t = Tensor(self.expt, requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of Exp
        self.t.backward(dLdf * self.expt)
        pass

def exp(t):
    return Exp().forward(t)


class Log(Function):
    
    def forward(self, t):
        super().forward(t)
        self.t = t
        requires_grad = t.requires_grad
        t = Tensor(np.log(t.data), requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of Log
        self.t.backward(dLdf / self.t.data)
        pass

def log(t):
    return Log().forward(t)


class Sum(Function):
    
    def forward(self, t, dim=0):
        super().forward(t)
        self.t = t
        self.dim = dim
        requires_grad = t.requires_grad
        t = Tensor(np.sum(t.data, axis=dim), requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of Sum
        dLdT = np.expand_dims(dLdf, axis = self.dim)
        self.t.backward(np.repeat(dLdT, self.t.data.shape[self.dim], axis = self.dim))
        pass


class Mean(Function):
    
    def forward(self, t):
        super().forward(t)
        self.t = t
        requires_grad = t.requires_grad
        t = Tensor(t.data.mean(), requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of Mean
        self.t.backward(np.ones_like(self.t.data) / self.t.data.size)

        # Erklärung: Da der Gradient eine Art gewichtung darstellt, und bei einem Mittelwert die
        # Gewichtung jedes Elemts die gleich ist, ist der grad nicht vom Wert sonder nur
        # von der Anzahl der Elemente der Matrix abhänig. Bei vier Elementen sind das 25% pro Element 
        pass


class GetItem(Function):
    
    def forward(self, t, index):
        super().forward(t)
        self.t = t
        self.index = index
        requires_grad = t.requires_grad
        t = Tensor(t.data[index], requires_grad=requires_grad)
        t.grad_fn = self
        return t
    
    def backward(self, dLdf):
        # implement the backward pass of GetItem
        pass

