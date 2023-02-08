

import torch
import torch.nn as nn
import torch.nn.functional as F



# Create a random softmax with pytorch
random = torch.rand(1, 10)
# Make sure it sums to 1
random = random / random.sum()
# Call softmax on it with pytorch
softmax = F.softmax(random, dim=1)
softmax = softmax/4 
# Test if it still sums to 1
print(softmax.sum())