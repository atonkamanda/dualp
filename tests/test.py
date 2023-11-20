import torch
import torch.nn.functional as F

softmax1 = torch.tensor([0.06, 0.02, 0.72, 0.05, 0.02, 0.05, 0.01, 0.01, 0.01, 0.05])
softmax2 = torch.tensor([0.02, 0.01, 0.59, 0.03, 0.01, 0.03, 0.01, 0.01, 0.01, 0.04])

kl_divergence = F.kl_div(softmax1.log_softmax(dim=0), softmax2.softmax(dim=0), reduction='sum')

print(kl_divergence.item())
