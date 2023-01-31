
import torch
from packaging import version
from torch import nn

from torchrl.envs.utils import step_mdp
from torchrl.modules.distributions import NormalParamWrapper
from torchrl.modules.models.models import MLP
#from torchrl.modules.tensordict_module.common import SafeModule
#from torchrl.modules.tensordict_module.sequence import SafeSequential

class ConvEncoder(nn.Module):
    """Convd2d encoder network.

    Takes an pixel observation and encodes it into a latent space.

    Args:
        depth (int, optional): Number of hidden units in the first layer.
            Defaults to 32.
    """

    def __init__(self, depth=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d(depth, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(depth, depth * 2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride=2),
            nn.ReLU(),
        )

    def forward(self, observation):
        *batch_sizes, C, H, W = observation.shape
        if len(batch_sizes) == 0:
            end_dim = 0
        else:
            end_dim = len(batch_sizes) - 1
        observation = torch.flatten(observation, start_dim=0, end_dim=end_dim) # Flatten the batch dimension
    
        obs_encoded = self.encoder(observation)
        latent = obs_encoded.reshape(*batch_sizes, -1)
        return latent

# Test the encoder with a random input
if __name__ == "__main__":
    encoder = ConvEncoder()
    obs = torch.randn(3, 3, 64, 64)
    latent = encoder(obs)
    print(latent.shape)