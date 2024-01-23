from torch.nn import Module
import torch

class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
        super().__init__()
        raise Exception("How was this built?")
        self.layer = torch.nn.GroupNorm(num_groups=num_groups, 
                                        num_channels=num_channels, 
                                        eps=eps, 
                                        affine=affine, 
                                        device=device, 
                                        dtype=dtype)
        self.use_bias = True
        
    def forward(self, x, save_list=None, mean_as_contribution=True):

        if len(x.shape) == 5:

            if self.use_bias:
                s, b, c, w, h = x.shape

                dim1, dim2 = self.layer.num_groups, c//self.layer.num_groups

                # Sources X Batch X Groups X GroupSize X H X W
                a = x.clone().reshape([s, b, dim1, dim2, w, h])

                # 1 X BATCH X Groups X C//GROUPS X H X W
                x_total = x.sum(0, keepdim=True).reshape([b, dim1, dim2, w, h])

                # 1 X BATCH X Groups X 1 X 1 X 1
                var, mu = torch.var_mean(x_total, dim=[-1,-2,-3], unbiased=False, keepdim=True)
                std = (var+self.layer.eps).sqrt()

                if mean_as_contribution:
                    mu = torch.mean(a, dim=[-1,-2,-3], keepdim=True)
                    a -= mu
                else:
                    a[0] -= mu

                a /= std.unsqueeze(0)

                # 1 X BATCH X C X H X W
                a = a.reshape([s,b,c,w,h])

                a *= self.layer.weight.reshape([1,1,-1,1,1])
                a[0] += self.layer.bias.reshape([1,-1,1,1])
                a = a.reshape([s,b,c,w,h])

            else:

                s, b, c, w, h = x.shape

                dim1, dim2 = self.layer.num_groups, c//self.layer.num_groups

                # Sources X Batch X Groups X GroupSize X H X W
                a = x.clone().reshape([s, b, dim1, dim2, w, h])

                # 1 X BATCH X Groups X C//GROUPS X H X W
                x_total = x.sum(0, keepdim=True).reshape([b, dim1, dim2, w, h])

                # 1 X BATCH X Groups X 1 X 1 X 1
                var, mu = torch.var_mean(x_total, dim=[-1,-2,-3], unbiased=False, keepdim=True)
                std = (var+self.layer.eps).sqrt()

                mu = torch.mean(a, dim=[-1,-2,-3], keepdim=True)
                a = a - mu

                a = a / (std.unsqueeze(0)+10e-30)

                # 1 X BATCH X C X H X W
                a = a.reshape([s,b,c,w,h])

                a = a * self.layer.weight.reshape([1,1,-1,1,1])

                frac = a.abs() / (a.abs().sum(0, keepdim=True) + 10e-30)

                a = a + frac * self.layer.bias.reshape([1,-1,1,1])
                a = a.reshape([s,b,c,w,h])
        else:
            a = self.layer(x)
        
        if save_list is not None:
            save_list.append(["group_norm", a.clone().detach()])
        return a

    def test(self, x):
        return self.layer(x.sum(0))
    


if __name__ == "__main__": 
    test = GroupNorm(4, 128, eps=1e-05, affine=True, device=None, dtype=None)

    for i in range(25):
        input = torch.randn((3,10,128,32,32), dtype=torch.double)*10
        test.double()
        b = test.test(input)
        a = test(input).sum(0)
        print(f"{i}:  {(a-b).abs().mean()}")