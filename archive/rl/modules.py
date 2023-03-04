import torch 
import torch.nn as nn
import torch.nn.functional as F
# Import independent from torch distribution 
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.distributions import Independent
# Import tensor from torch
from torch import Tensor
# Import optional from typing
from typing import Optional
import numpy as np
class ConvEncoder(nn.Module):
    def __init__(self, act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.act =  act
        self.fc = nn.Linear(64 * 64, 256)
        x = self.conv3(x)
        x = self.act(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc = nn.Linear(256, 64 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4, padding=2)
        self.act = act
        
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = x.view(x.size(0), 64, 8, 8)
        x = self.deconv1(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.act(x)
        x = self.deconv3(x)
        return x
    
class ConvDecoder(nn.Module):
    def __init__(self, act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc = nn.Linear(256, 64 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4, padding=2)
        self.act = act
        
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = x.view(x.size(0), 64, 8, 8)
        x = self.deconv1(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.act(x)
        x = self.deconv3(x)
        return x

class Actor(nn.Module):
    def __init__(self, action_shape,act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, action_shape)
        self.act = act
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
class Critic(nn.Module):
    def __init__(self,act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = act
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Reward_predictor(nn.Module):
    def __init__(self,act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = act
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class RSSM(nn.Module):
    def __init__(self,
                 action_shape: tuple,
                 stoch: int = 30,
                 deter: int = 200,
                 hidden: int = 200,
                 embed: int = 1024,
                 act: nn.Module = nn.ELU, cfg=None):
        super().__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.activation = act
        self.stoch_size = stoch
        self.deter_size = deter
        self.hidden_size = hidden
        self.embed_size = embed
        # two "hidden" has different meanings: one is the standard usage, the other means RNN's hidden's size
        self.cell = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.deter_size)
        self.act = act()
        self.rssm_std = 0.1 if cfg is None else cfg.rssm_std 

        # Flatten action space 
        action_shape = np.prod(action_shape)
        # action + state -> GRU input
        rnn_input_size = stoch + action_shape
        self.fc_input = nn.Sequential(nn.Linear(rnn_input_size, hidden), self.act)
        # deter state -> next prior
        self.fc_prior = nn.Sequential(
            nn.Linear(deter, hidden), self.act,
            nn.Linear(hidden, 2 * stoch)
        )
        # deter state + image -> next posterior
        self.fc_post = nn.Sequential(
            nn.Linear(deter + embed, hidden), self.act,
            nn.Linear(hidden, 2 * stoch)
        )

    def initial(self, batch_size: int):
        return dict(mean=torch.zeros(batch_size, self.stoch_size, device=self.device),
                    std=torch.zeros(batch_size, self.stoch_size, device=self.device),
                    stoch=torch.zeros(batch_size, self.stoch_size, device=self.device),
                    deter=torch.zeros(batch_size, self.deter_size, device=self.device))
    
    def get_feat(self, state: dict):
        return torch.cat([state['stoch'], state['deter']], -1) # (B, stoch + deter) -1 for last dim

    def get_dist(self, state: dict):
        return Independent(Normal(state['mean'], state['std']), 1) 

    def observe(self, embed: Tensor, action: Tensor, state: Optional[Tensor] = None):
        """
        Compute prior and posterior given initial prior, actions and observations.  

        Args:
            embed: (B, T, D) embeded observations
            action: (B, T, D) actions. Note action[t] leads to embed[t]
            state: (B, D) or None, initial state
        Returns:
            post: dict, same key as initial(), each (B, T, D)
            prior: dict, same key as initial(), each (B, T, D)

        Here "state" is params (mean / std / stoch / deter) of a distribution (prior / posterior)
        """
        B, T, _ = action.size()
        if state is None:
            state = self.initial(B)
        post_list = []
        prior_list = []
        for t in range(T):
            post_state, prior_state = self.obs_step(state, action[:, t], embed[:, t])
            prior_list.append(prior_state)
            post_list.append(post_state)
            state = post_state
        prior = {k: torch.stack([state[k] for state in prior_list], dim=1) for k in prior_list[0]}
        post = {k: torch.stack([state[k] for state in post_list], dim=1) for k in post_list[0]}
        return post, prior

    def imagine(self, action: Tensor, state: Optional[Tensor] = None):  # used in video logging
        """
        Compute priors given initial prior and actions.

        Almost the same as observe so nothing special here
        Args:
            action: (B, T, D) actions. Note action[t] leads to embed[t]
            state: (B, D) or None, initial state
        Returns:
            prior: dict, same key as initial(), each (B, T, D)
        """
        B, T, D = action.size()
        if state is None:
            state = self.initial(B)
        assert isinstance(state, dict)
        prior_list = []
        for t in range(T):
            state = self.img_step(state, action[:, t])
            prior_list.append(state)
        prior = {k: torch.stack([state[k] for state in prior_list], dim=1) for k in prior_list[0]}
        return prior

    def obs_step(self, prev_state: Tensor, prev_action: Tensor, embed: Tensor):
        """
        Compute next prior and posterior given previous prior and action
        Args:
            prev_state: (B, D) or None, initial state
            prev_action: (B,  D) actions. 
            embed: (B,  D) embeded observations
        Returns:
            post: dict, same key as initial(), each (B, D)
            prior: dict, same key as initial(), each (B, D)
        """
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self.fc_post(x)
        mean, std = x.chunk(2, dim=-1)
        std = F.softplus(std) + self.rssm_std
        # stoch = self.get_dist(dict(mean=mean, std=std)).rsample()  # slow
        stoch = torch.randn_like(std) * std + mean  # posterior sample
        post = dict(mean=mean, std=std, stoch=stoch, deter=prior['deter'])  # prior['deter'] == post['deter']
        return post, prior

    def img_step(self, prev_state: Tensor, prev_action: Tensor):
        """        print(prev_action.device)
        print(prev_state['stoch'].device)ct, same key as initial(), each (B, D)
            prior: dict, same key as initial(), each (B, D)
        """
        rnn_input = torch.cat([prev_state['stoch'], prev_action], dim=-1)
        rnn_input = self.fc_input(rnn_input)
        x = deter = self.cell(rnn_input, prev_state['deter'])
        x = self.fc_prior(x)
        mean, std = x.chunk(2, dim=-1)
        std = F.softplus(std) + self.rssm_std
        # stoch = self.get_dist(dict(mean=mean, std=std)).rsample()  # slow
        stoch = torch.randn_like(std) * std + mean  # prior sample
        prior = dict(mean=mean, std=std, stoch=stoch, deter=deter)
        return prior

# Script version
if __name__ == '__main__':
    
    def test_encoder():
        encoder = ConvEncoder()
        x = torch.randn(1,3,64,64)
        z = encoder(x)
        print(z.shape)
        
    def test_decoder():
        decoder = ConvDecoder()
        z= torch.randn(1,256)
        x = decoder(z)
        print(x.shape)

    def test_actor():
        actor = Actor(4)
        z = torch.randn(1,256)
        a = actor(z)
        
    def test_rssm():
        T = 10
        B = 4
        D = 5
        action = torch.randn(B, T, D)
        embed = torch.randn(B, T, 1024)
        #action_space = spaces.Box(low=-1, high=1, shape=(D,))
        action_shape = np.array([D])
        rssm = RSSM(action_shape).to('cpu')
        prior = rssm.imagine(action)
        prior, post = rssm.observe(embed, action)
        print(prior['deter'].size())
        print(post['deter'].size())

    #test_encoder()
    #test_decoder()
    #test_actor()
    #test_critic()
    test_rssm()
