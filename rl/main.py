# Python
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime
import random 
from utils import Logger, set_seed
from replay_buffer import ReplayBuffer
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
import gymnasium as gym
from gym.wrappers import record_video,record_episode_statistics
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
    record_episode_statistics : bool = False 
    log_every : int = 1000
    print_every : int = 10
    
    
    
    # Task hyperparameters 
    train_env : str = 'LunarLander-v2'
    task : str = 'run'
    epoch : int = 10 # The number of update 
    total_timesteps : int = 20000
    
    # Model hyperparameters
    batch_size : int = 64
    policy_lr : float = 1e-3
    value_lr : float = 1e-3
    
class EnvManager():
    """ Take an env and apply wrapper to it"""
    def __init__(self,config : Config):
        super().__init__()
        self.c = config
    
    def apply_wrappers(self,env):
        if self.c.record_episode_statistics:
            env = record_episode_statistics.RecordEpisodeStatistics(env)
        if self.c.record_video:
            env = record_video.RecordVideo(env,video_folder='./videos',episode_trigger = lambda x: x % 10==0)
            print('Video recording enabled')
        return env
    
    def create_envs(self,env_name : str):
        env = gym.make(env_name, render_mode="rgb_array")
        env = self.apply_wrappers(env)
        # Print observation and action space
        print('Observation space : ',env.observation_space.shape)
        print('Action space : ',env.action_space.shape)
        return env

class Agent(nn.Module):
    def __init__(self,config : Config,obs_space ,act_space ):
        
        # Initialize the agent hyperparameters
        super().__init__() 
        self.c = config
        self.obs_size = 8
        self.act_size = 4
        self.state_size = 10
        self.encoder = nn.Sequential(
                        nn.Linear(self.obs_size,self.state_size),
                        nn.ReLU())
        
        # Initialize the neural networks modules
        self.actor = nn.Sequential(
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



    def update(self,replay_buffer) -> None:  
        data = replay_buffer.sample(self.c.batch_size)
        latent = self.encoder(data['obs'])
        value = self.critic(latent)
        
class Trainer:
    def __init__(self, config:Config):
        self.c = config
        set_seed(self.c.seed,self.c.device)
        self.env = EnvManager(self.c).create_envs(self.c.train_env)
        
        self.agent = Agent(self.c,self.env,self.env)
        self.logger = Logger()
        #self.replay_buffer = ReplayBuffer(10000)
        self.ep_reward_list = []
 

        if self.c.device == 'auto':
            self.c.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
    def calculate_advantage(rewards, values, gamma, lambda_value):
        """
        Calculate the advantage for a given set of rewards, values and hyperparameters

        Parameters:
        rewards (list): a list of rewards for each time step
        values (list): a list of estimated state-values for each time step
        gamma (float): discount factor
        lambda_value (float): GAE lambda hyperparameter

        Returns:
        list: the advantage for each time step
        """
        T = len(rewards)
        advantages = []
        advantage = 0
        for t in range(T - 1, -1, -1):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            advantage = delta + gamma * lambda_value * advantage
            advantages.insert(0, advantage)
        return advantages

            
    def collect_experience(self,n_steps : int = 1000):
        train_data = [[] for _ in range(5)] # obs , action, reward, values , act_log_probs
        obs, info  = self.env.reset()
        ep_reward = 0 
        for t in range(n_steps):
            obs = torch.tensor(obs).float().to(self.c.device)
            z = self.agent.encoder(obs)
            policy = self.agent.actor(z) 
            value = self.agent.critic(z)
            dist = Categorical(policy)
            act = dist.sample()
            next_obs, rew, terminated, truncated, info = self.env.step(act.item())
            ep_reward += rew
          
            if terminated or truncated:
                print(f"Episode reward : {ep_reward}")
                # print episode lenght
                print(f"Episode length : {t}")
                for i, data in enumerate([obs,act,rew,value,dist.log_prob(act)]): 
                    train_data[i].append(data)
                break
                observation, info = self.env.reset()
            obs = next_obs
        self.env.close()
        return train_data

    def train(self):
        test = self.collect_experience(1000)
        print(test)
        #for update in range(100):

        
        
    def eval(self):
        pass 
    
    def compute_true_returns(self,rewards):
        true_returns = []
        true_return = 0
        discount_factor = 1
        for reward in rewards[::-1]:
            true_return = reward + discount_factor * true_return
            true_returns.append(true_return)
            discount_factor = discount_factor * 0.99
        return true_returns[::-1]
    
            
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