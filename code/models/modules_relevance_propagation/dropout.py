import torch

class Dropout(torch.nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()

        self._warning_printed = False
        self.rate = p

    @staticmethod
    def from_torch(layer):
        assert type(layer) is torch.nn.Dropout
        l = Dropout(p=layer.p)
        
        return l

    def to_torch(self):
        return torch.nn.Dropout(p=self.rate)

    def forward(self, x):
        print("WARNING DROPOUT DEACTIVATED ")
        return x
    
        if not self._warning_printed:
            print("Warning Dropout Active Status: ", self.training)
            self._warning_printed = True

        if not self.training:
            return x

        if len(x.shape) == 5 or len(x.shape) == 3:
            mask = torch.bernoulli(self.rate * torch.ones([1,*x.shape[1:]], device=x.device)) == 0
        else:
            mask = torch.bernoulli(self.rate * torch.ones_like(x, device=x.device)) == 0

        out = torch.where(mask, x, torch.zeros_like(x))

        return out