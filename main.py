# Python
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime
import random 
from utils import Logger
# Torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions.categorical import Categorical
# ML
import matplotlib.pyplot as plt
from omegaconf import OmegaConf,DictConfig
import hydra
import numpy as np
import pandas as pd
import cv2
# Env 
from torchrl.envs.libs.dm_control import DMControlEnv
import gym
@dataclass
class Config:
    
    # Reproductibility and hardware 
    seed : int = 0
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    load_agent : bool = False
    job_num : int = 0

    # Logging
    logdir : str = pathlib.Path.cwd() / 'logs'
    record_video : bool = False
    log_every : int = 1000
    print_every : int = 10
    
    
    
    # Task hyperparameters 
    train_env : str = 'acrobot'
    task : str = 'run'
    epoch : int = 10 # The number of update 
    total_timesteps : int = 20000
    
    # Model hyperparameters
    batch_size : int = 64
    can_sleep : bool = True
    sleep_itr : int = 10000
    wake_lr : float = 0.02
    sleep_lr : float = 0.001    
    
class EnvManager():
    """ Take an env and apply wrapper to it"""
    def __init__(self,config : Config):
        super().__init__()
        self.c = config
    
    def apply_wrappers(self,env):
        if self.c.record_video:
            print('Video recording enabled')
        return env
    
    def create_envs(self,env_name : str):
        env = DMControlEnv(env_name, "swingup",from_pixels=True, pixels_only=True)
        env.set_seed(self.c.seed)
        return env

class Agent(nn.Module):
    def __init__(self,config : Config,obs_space ,act_space ):
        
        # Initialize the agent hyperparameters
        super().__init__() 
        self.c = config
        #self.obs_size = obs_space.shape[0]
        #self.act_size = act_space.n
        #self.state_size = self.c.state_size
        self.encoder = None
        
        # Initialize the neural networks modules

        """self.actor = nn.Sequential(
                        nn.Linear(self.state_size,self.act_size),
                        nn.Softmax(dim=-1))
        self.critic = nn.Sequential(
                        nn.Linear(self.state_size,1))
        
        # Initialize the parameters 
        self.policy_params = list(self.encoder.parameters()) + list(self.actor.parameters())
        self.value_params = list(self.encoder.parameters()) + list(self.critic.parameters())

        # Initialize them with rmsprop
        self.policy_optimizer = optim.RMSprop(self.policy_params, lr=self.c.policy_lr)
        self.value_optimizer = optim.RMSprop(self.value_params, lr=self.c.value_lr)
    """


    def update(self) -> None:  
        pass
        
class Trainer:
    def __init__(self, config:Config):
        self.c = config
        self.seed = self.c.seed
        self.env = EnvManager(self.c).create_envs(self.c.train_env)
        #self.test_env = EnvManager(self.c).create_envs(self.c.test_env)
        self.agent = Agent(self.c,self.env,self.env)
        self.logger = Logger()
        self.ep_reward_list = []
 

        if self.c.device == 'auto':
            self.c.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
    def set_seed(self,seed : int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.c.device == 'cuda':
            torch.cuda.manual_seed_all(seed)
    def collect_experience(self):
        pass   

    def train(self):
        tensordict = self.env.reset()
        print("result of reset: ", tensordict)
        plt.imshow(tensordict.get("pixels").numpy())
        plt.show()
        self.env.close()
        #self.logger.write_video('test.mp4',tensordict["pixels"].numpy(),fps=60)
        
        
    def eval(self):
        pass 
  
    
            
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # Load default config
    default_config = OmegaConf.structured(Config)
    # Merge default config with run config, run config overrides if there is a conflict
    config = OmegaConf.merge(default_config, cfg)
    #OmegaConf.save(config, 'config.yaml') 
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #job_num = hydra_cfg.job.num
    #print(f'Hydra job number: {job_num}')
    #config.job_num = job_num
    trainer = Trainer(config)
    trainer.train()
    
    
if __name__ == '__main__':
    main()
    print('Finished correctly')