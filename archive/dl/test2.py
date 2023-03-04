
import torch
import torch.nn as nn

class CNN_MNIST_Dual(nn.Module):

    def __init__(self):
        super(CNN_MNIST_Dual, self).__init__()
        # CNN for MNIST dataset
        
        self.encoder = nn.Sequential(
                        nn.Conv2d(1, 10, kernel_size=5),
                        nn.ReLU(),
                        nn.MaxPool2d(2)) 
        self.control = nn.Sequential(
                        nn.Conv2d(10, 20, kernel_size=5), # Shape : 20 x 8 x 8
                        nn.ReLU(),
                        nn.MaxPool2d(2), # Shape : 20 x 4 x 4
                        nn.Linear(320, 50), # Shape : 50
                        nn.ReLU(),
                        nn.Linear(50, 10))
        
        self.habitual = nn.Sequential(
                        nn.Linear(1440, 10))
        
                        
    
    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        # Flatten the output
        x = x.view(-1, 1440)
        print(x.shape)
        return x 
    
# Test on random data
model = CNN_MNIST_Dual()
data = torch.randn(1, 1, 28, 28)
model(data)