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
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(-3))

        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(-3)) # 12 x 12 x 64

        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(2 * ((input_shape[0]-4) // 2) * ((input_shape[1]-4) // 2) * 64, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x1_in, x2_in):
        
        x1 = self.branch1(x1_in)
        x2 = self.branch2(x2_in)
        
        x = torch.cat([x1,x2], -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = x
        return output
