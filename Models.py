import torch.nn as nn
import torch.nn.functional as F

### Define the SCAE as per the paper, although imo, it is very basic
class StackedConvAutoencoder(nn.Module):
    def __init__(self):
        super(StackedConvAutoencoder, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)  # Output: 32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # Output: 16x16
        
        # Decoder
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)  # Output: 16x16
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 32x32
        
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x, indices = self.pool(x)
        
        # Decoder
        x = self.unpool(x, indices)
        x = F.relu(self.conv2(x))
        return x