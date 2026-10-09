import torch
import torch.nn as nn

class RLmodel(nn.Module):
    def __init__(self, gridsize, input_channels = 2, n_actions = 4):
        super().__init__()
        size = gridsize + 2
        self.conv = nn.Sequential(nn.Conv2d(input_channels, 32, 3, padding = 1), nn.ReLU(),
                                   nn.Conv2d(32, 64, 3, padding = 1), nn.ReLU(),
                                     nn.Conv2d(64, 16, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(16*size*size,256), nn.ReLU(),nn.Linear(256, n_actions))
    def forward(self,x):
        return self.fc(self.conv(x))

