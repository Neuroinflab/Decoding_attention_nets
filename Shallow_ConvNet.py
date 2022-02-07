import torch
import torch.nn as nn
from einops import rearrange

class Shallow(nn.Module):
    
    def __init__(self):
        
        
        original = 250 # Fs [Hz]
        my = 400 # Fs [Hz]
        prop = my / original
        
        self.conv1 = nn.Conv2d(1, 40, (1, int(25 * prop)))
        
        self.conv2 = nn.Conv2d(40, 40, (19, 1))
        
        self.avr = nn.AvgPool2d((1, int(75 * prop)), stride=(1, int(15 * prop)))
        
        self.lin = nn.Linear(3080, 1)

    def forward(self, x):
        
        z = self.conv1(x)
        z = self.conv2(z)
        z = torch.square(z)
        z = rearrange(z, 'b c 1 w -> b c w')
        z = self.avr(z)
        z = torch.log(z)
        z = rearrange(z, 'b c w -> b (c w)')
        z = self.lin(z)
        z = rearrange(z, 'b 1 -> b')
        z = torch.sigmoid(z)
        return z
    
    