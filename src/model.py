import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2, padding=0)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2, padding=0)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.act5 = nn.ReLU()
        self.drop_out1 = nn.Dropout()
        self.fc1 = nn.Linear(in_features=27456, out_features=100)
        self.act6 = nn.ReLU()
        self.drop_out2 = nn.Dropout()
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.act7 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=50, out_features=10)
        self.act8 = nn.ReLU()
        self.output = nn.Linear(in_features=10, out_features=1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.drop_out1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act6(x)
        x = self.drop_out2(x)
        x = self.fc2(x)
        x = self.act7(x)
        x = self.fc3(x)
        x = self.act8(x)
        x = self.output(x)
        x = x.view(-1)
        return x
    
if __name__ == "__main__":
    model = CNN()
    fake_data = torch.rand((8, 3, 160, 320))
    output = model(fake_data)
    print(output.shape)