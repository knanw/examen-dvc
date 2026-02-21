import torch.nn as nn
# Architektur des Neuronalen Netzes

class SilicaPredictor(nn.Module):
    def __init__(self, input_size):
        super(SilicaPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)