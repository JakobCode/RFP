import torch

class BatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):        
        super().__init__()

        self.visit = 0
        self.eps = eps
        self.num_features = num_features
        self.momentum=momentum
        self.affine=affine
        self.track_running_stats=track_running_stats

        self.layer = torch.nn.BatchNorm2d(num_features=num_features, 
                                       eps=eps,
                                       momentum=self.momentum,
                                       affine=self.affine,
                                       track_running_stats=self.track_running_stats)
        self.use_bias = True

        # self.last_value_in = 0
        # self.last_value_out = 0
        #self.layer.running_mean += torch.rand([num_features])
        #self.layer.running_var += 0.1 * torch.rand([num_features])

    @staticmethod
    def from_torch(layer):
        assert type(layer) is torch.nn.BatchNorm2d
        l = BatchNorm2d(num_features=layer.num_features,
                        eps=layer.eps, 
                        momentum=layer.momentum, 
                        affine=layer.affine, 
                        track_running_stats=layer.track_running_stats)
        
        l.layer = layer

        return l

    def to_torch(self):
        return self.layer
        

    def forward(self, x):
        if len(x.shape) == 5:

            # self.last_value_in = x.sum(0).detach()

            out = x + 0

            if self.use_bias:
                out[0] = out[0] - self.layer.running_mean.reshape((1,self.num_features,1,1))    
                out = self.layer.weight.reshape((1,1,self.num_features,1,1)) * (out/(torch.sqrt(self.layer.running_var + self.eps)).reshape((1,1,self.num_features,1,1)))

                out[0] += self.layer.bias.reshape((1,self.num_features,1,1))

            else:
                raise Exception()
                frac = x.abs() / (x.abs().sum(0,keepdim=True) + 10e-30)

                out = out - frac * self.layer.running_mean.reshape((1,self.num_features,1,1))    
                out = self.layer.weight.reshape((1,1,self.num_features,1,1)) * (out/torch.sqrt(self.layer.running_var + self.eps).reshape((1,1,self.num_features,1,1)))

                out = out +  frac * self.layer.bias.reshape((1,self.num_features,1,1))

            # self.last_value_out = out.sum(0).detach()
        else:
            out = x + 0
            # self.last_value_in = x.detach()

            out = self.layer(x)       
            # self.last_value_out = out.detach()

        return out