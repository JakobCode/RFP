"""
Relevance Forward Propagation (RFP) functionalities for ReLU and LeakyReLU
"""
import torch

class ReLU(torch.nn.Module):
    def __init__(self) -> None:
        """
        Initialize Relevance Forward Propagation ReLU
        """
        super().__init__()

    def forward(self, x):
        """
        Forward input sample x

        Arguments: 
        x       torch.Tensor        input to be propagated

        Output
        out     torch.Tensor        processed output
        """

        if len(x.shape) == 5 or len(x.shape) == 3:
            mask = torch.sum(x, dim=0, keepdim=True) >= 0
        else:
            mask = x >= 0

        out = mask * x

        return out

    def to_torch(self):
        """
        Returns basic PyTorch implementation of the ReLU Module
        """
        return torch.nn.ReLU()     

    @staticmethod
    def from_torch(layer):
        """
        Build relevance forward propagation ReLU from ReLU Module
        """
        assert type(layer) is torch.nn.ReLU

        return ReLU()          

class LeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.1) -> None:
        """
        Initialize Relevance Forward Propagation LeakyReLU
        """
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        Forward input sample x

        Arguments: 
        x       torch.Tensor        input to be propagated

        Output
        out     torch.Tensor        processed output
        """
        if len(x.shape) == 5 or len(x.shape) == 3:
            mask = torch.sum(x, dim=0, keepdim=True) >= 0
        else:
            mask = x >= 0

        out = torch.where(mask, x, self.negative_slope * x)

        return out

    def to_torch(self):
        """
        Returns basic PyTorch implementation of the LeakyReLU Module
        """
        return torch.nn.LeakyReLU(negative_slope=self.negative_slope)   

    @staticmethod
    def from_torch(layer):
        """
        Build relevance forward propagation LeakyReLU from LeakyReLU Module
        """
        assert type(layer) is torch.nn.LeakyReLU

        return LeakyReLU(negative_slope=layer.negative_slope)   



### Adjust the code below
class GELU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.gelu_mode = None
        self.use_bias = True
        
    def forward(self, x):
        """
        Forward input sample x

        Arguments: 
        x       torch.Tensor        input to be propagated

        Output
        out     torch.Tensor        processed output
        """
        if len(x.shape) == 5 or len(x.shape) == 3:

            mask_r = torch.sum(x, dim=0, keepdim=True) >= 0

            out = torch.where(mask_r, x, torch.zeros_like(x))
            
            if self.gelu_mode == "gelu_opt1":
                # Option 1 (Split differences weighted)
                f = x.abs() / (x.abs().sum(0, keepdims=True) + 10e-30)
                x_t = self.gelu(x.sum(0, keepdim=True))
                out = out + f * (x_t-out.sum(0, keepdim=True))
            elif self.gelu_mode == "gelu_opt2":
                assert self.use_bias
                # Option 2 (Difference as bias)
                x_t = self.gelu(x.sum(0))
                out[0] +=  x_t-out.sum(0)
            else:
                raise Exception(self.gelu_mode)

        else:
            out = self.gelu(x)

        return out

        

