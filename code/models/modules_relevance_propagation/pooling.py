"""
Relevance Forward Propagation (RFP) functionalities for AdaptiveAvgPool2d, AvgPool2d and MaxPool2d
"""

import torch

class AdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size) -> None:
        """
        Initialize Relevance Forward Propagation AdaptiveAvgPool2d
        """
        super().__init__()

        self.output_size = output_size
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    @staticmethod
    def from_torch(layer):
        assert type(layer) is torch.nn.AdaptiveAvgPool2d
        l = AdaptiveAvgPool2d(layer.output_size)
        l.pool = layer

        return l

    def to_torch(self):
        return torch.nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, x):
        """
        Forward input sample x

        Arguments: 
        x       torch.Tensor        input to be propagated

        Output
        out     torch.Tensor        processed output
        """
        if len(x.shape)==5:

            num_sources = x.shape[0]
            num_batch = x.shape[1]

            out = torch.reshape(x, (num_sources*num_batch, *x.shape[2:]))
            out = self.pool(out)
            out = torch.reshape(out, (num_sources, num_batch, *out.shape[1:]))

        else:
            out = self.pool(x)

        return out

class AvgPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2) -> None:
        """
        Initialize Relevance Forward Propagation AvgPool2d
        """
        super().__init__()
        
        self.pooling = torch.nn.AvgPool2d(kernel_size=kernel_size,stride=stride)

    @staticmethod
    def from_torch(layer):
        assert type(layer) is torch.nn.AvgPool2d
        l = AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride)
        l.pooling = layer

        return l

    def to_torch(self):
        return torch.nn.AvgPool2d(kernel_size=self.pooling.kernel_size, stride=self.pooling.stride)
    
    def forward(self, x):
        """
        Forward input sample x

        Arguments: 
        x       torch.Tensor        input to be propagated

        Output
        out     torch.Tensor        processed output
        """

        if len(x.shape)==5: 
            num_sources = x.shape[0]
            num_batch = x.shape[1]

            out = torch.reshape(x, (num_sources*num_batch, *x.shape[2:]))
            out = self.pooling(out)
            out = torch.reshape(out, (num_sources, num_batch, *out.shape[1:]))  

        else:
            out = self.pooling(x)

        return out

class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):      
        """
        Initialize Relevance Forward Propagation MaxPool2d
        """  
        super().__init__()

        self.kernel_size = kernel_size
        
        if isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size)
        else: 
            self.kernel_size = kernel_size

        if isinstance(padding,int):
            self.padding = (padding,padding)
        else: 
            self.padding = padding

        self.stride = stride if stride is not None else kernel_size
        if isinstance(self.stride, int): self.stride = [self.stride, self.stride]
        
        self.padding = padding
        self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, 
                                      stride=self.stride, 
                                      padding=self.padding)

    @staticmethod
    def from_torch(layer):
        assert type(layer) is torch.nn.MaxPool2d
        l = MaxPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
        l.layer = layer

        return l

    def to_torch(self):
        return torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
    

    def forward(self, x):
        """
        Forward input sample x

        Arguments: 
        x       torch.Tensor        input to be propagated

        Output
        out     torch.Tensor        processed output
        """
                
        if len(x.shape) == 5:
            xUfd1 = self.unfold(x.sum(dim=0)).reshape((x.shape[1], x.shape[2], self.kernel_size[0]*self.kernel_size[1], -1))
            xUfd2 = self.unfold(x.reshape((x.shape[0]*x.shape[1],*x.shape[2:]))).reshape((x.shape[0], x.shape[1], x.shape[2], self.kernel_size[0]*self.kernel_size[1], -1))

            _, xUfd_idx = torch.max(xUfd1.unsqueeze(0), -2, keepdim=True)

            xUfd_idx = xUfd_idx.repeat([x.shape[0],1,1,1,1])
            out = torch.gather(xUfd2, dim=-2, index=xUfd_idx).reshape((x.shape[0],x.shape[1],x.shape[2], x.shape[-2]//self.stride[0],x.shape[-1]//self.stride[1]))

        elif len(x.shape) == 4:
            out = self.layer(x)

        return out