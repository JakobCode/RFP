import torch


def view(self, *args):
    print("view view view")
    if len(args) == 2 and args[0] == self.shape[0] and args[1] == -1:
        args = [*self.shape[:2], -1]
    return self.view(*args)

def interpolate(x, size, mode='bilinear', align_corners=True):

    if len(x.shape) == 5: 
        size = size[-2:]
        s, b = x.shape[:2]
        x = x.reshape((s * b, *x.shape[2:]))

        x = torch.nn.functional.interpolate_orig(x, size=size, mode=mode, align_corners=align_corners)

        x = x.reshape((s, b, *x.shape[1:]))

    else:
        x = torch.nn.functional.interpolate_orig(x, size=size, mode=mode, align_corners=align_corners)
        
    return x

def relu(x, inplace=False):
    if len(x.shape) == 5 or len(x.shape) == 3:
        mask = torch.sum(x, dim=0, keepdim=True) >= 0
    else:
        mask = x >= 0

    out = torch.where(mask, x, torch.zeros_like(x))

    return out


def flatten(x, start_dim=1):
    if start_dim == 1 or start_dim == -3:
        return torch.flatten_orig(x, start_dim=-3)

def cat(x, dim=0):
    if len(x[0].shape) == 5 and dim==1:
        dim=-3

    return torch.cat_orig(x, dim=dim)