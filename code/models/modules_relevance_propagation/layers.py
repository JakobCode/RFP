import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features : int, out_features : int, bias : bool=True) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_features=in_features, 
                                     out_features=out_features,
                                     bias=bias)

        self.use_bias = True
        # self.last_value_in = 0
        # self.last_value_out = 0

    @staticmethod
    def from_torch(layer):
        assert type(layer) is torch.nn.Linear
        l = Linear(in_features=layer.in_features, out_features=layer.out_features, bias=layer.bias is not None)
        l.layer = layer

        return l
    
    def to_torch(self):
        return self.layer

    def forward(self, x):

        if len(x.shape)==3: 
            # print("Linear in:", (x.sum(0)-# self.last_value_in).abs().mean().item(), (x.sum(0)-# self.last_value_in).abs().max().item())            
            # self.last_value_in = x.sum(0).detach()

            num_sources = x.shape[0]
            num_batch = x.shape[1]

            x = torch.reshape(x, (num_sources*num_batch, x.shape[-1]))
            
            x = self.layer(x)

            if self.layer.bias is not None: 
                x -= self.layer.bias

            x = torch.reshape(x, (num_sources, num_batch, x.shape[-1]))

            if self.layer.bias is not None: 
                if self.use_bias:
                    x[0] += self.layer.bias
                else:
                    frac = x.abs() / (x.abs().sum(0,keepdim=True) + 10e-30)
                    x = x + frac * self.layer.bias

            # print("Linear out:", (x.sum(0)-# self.last_value_out).abs().mean().item(), (x.sum(0)-# self.last_value_out).abs().max().item())            
            # self.last_value_out = x.sum(0).detach()
        else:

            # print("Linear in:", (x-# self.last_value_in).abs().mean().item(), (x-# self.last_value_in).abs().max().item())            
            # self.last_value_in = x.detach()
            x = self.layer(x)
            # print("Linear out:", (x-# self.last_value_out).abs().mean().item(), (x-# self.last_value_out).abs().max().item())            
            # self.last_value_out = x.detach()

        return x


class Conv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) -> None:
        super().__init__()

        self.layer = torch.nn.Conv2d(in_channels=in_channels, 
                                     out_channels=out_channels, 
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     padding=padding, 
                                     bias=bias, 
                                     groups=groups, 
                                     dilation=dilation,
                                     padding_mode=padding_mode,
                                     device=device,
                                     dtype=dtype)
        
        self.use_bias = True

        # self.last_value_in = 0
        # self.last_value_out = 0

    @staticmethod
    def from_torch(layer):
        assert type(layer) is torch.nn.Conv2d
        l = Conv2d(in_channels=layer.in_channels, 
                   out_channels=layer.out_channels, 
                   kernel_size=layer.kernel_size, 
                   stride=layer.stride, 
                   padding=layer.padding, 
                   dilation=layer.dilation, 
                   groups=layer.groups, 
                   bias=layer.bias is not None, 
                   padding_mode=layer.padding_mode, 
                   device=layer.weight.device, 
                   dtype=layer.weight.dtype)
        
        l.layer = layer

        return l

    def to_torch(self):
        return self.layer
    

    def forward(self, x, save_list=None):

        ## print("Conv2d in:", x.mean())
        if len(x.shape)==5: 

            # print("Conv2d in:", (x.sum(0)-# self.last_value_in).abs().mean().item(), (x.sum(0)-# self.last_value_in).abs().max().item())          
            # self.last_value_in = x.sum(0).detach()

            num_sources = x.shape[0]
            num_batch = x.shape[1]

            x = torch.reshape(x, (num_sources*num_batch, *x.shape[2:]))
            x = self.layer(x)
            if self.layer.bias is not None: 
                x -= self.layer.bias.unsqueeze(-1).unsqueeze(-1)

            x = torch.reshape(x, (num_sources, num_batch, *x.shape[1:]))
            if self.layer.bias is not None: 
                if self.use_bias:
                    x[0] += self.layer.bias.unsqueeze(-1).unsqueeze(-1)
                else:
                    frac = x.abs() / (x.abs().sum(0,keepdim=True) + 10e-30)
                    x = x + frac * self.layer.bias.unsqueeze(-1).unsqueeze(-1)

            # print("Conv2d out:", (x.sum(0)-# self.last_value_out).abs().mean().item(), (x.sum(0)-# self.last_value_out).abs().max().item())
            # self.last_value_out = x.sum(0).detach()

        else:
            # print("Conv2d in:", (x-# self.last_value_in).abs().mean().item(), (x-# self.last_value_in).abs().max().item())            
            # self.last_value_in = x.detach()

            x = self.layer(x)

            # print("Conv2d out:", (x-# self.last_value_out).abs().mean().item(), (x-# self.last_value_out).abs().max().item())
            # self.last_value_out = x.detach()


        return x


    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        return self.layer.load_state_dict(state_dict, strict)

    #def _save_to_state_dict(self, destination, prefix, keep_vars):
    #    return self.layer()._save_to_state_dict(destination, prefix, keep_vars)

class Identity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x): 
        return x

    def to_torch(self):
        return torch.nn.Identity()