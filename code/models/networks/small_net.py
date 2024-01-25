import torch

class SmallNet(torch.nn.Module):
    def __init__(self, input_shape=[28,28], num_classes=10):
        super(SmallNet, self).__init__()

        self.relu = torch.nn.ReLU()
        self.nclasses = num_classes
        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))

        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)) # 12 x 12 x 64

        self.fc1 = torch.nn.Linear(2 * ((input_shape[0]-4) // 2) * ((input_shape[1]-4) // 2) * 64, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x1_in, x2_in=None):
        
        if x2_in is None: 
            assert x1_in.shape[1] == 2
            x1_in, x2_in = x1_in[:,0], x1_in[:,1]

        x1 = torch.flatten(self.branch1(x1_in), start_dim=-3)
        x2 = torch.flatten(self.branch2(x2_in), start_dim=-3)
        
        x = torch.cat([x1,x2], -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = x
        return output
