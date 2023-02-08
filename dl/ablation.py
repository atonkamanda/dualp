import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pathlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from omegaconf import OmegaConf,DictConfig
import hydra
import pickle
import numpy as np
import pandas as pd
from utils import Logger,compare_beliefs, VariationalDropout
from termcolor import colored

# Class for the ablative study
from CNN_MNIST_dual import CNN_MNIST_Dual


@dataclass
class Config:
    
    # Reproductibility and hardware 
    seed : int = 0
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_model : bool = False
    job_num : int = 0

    # Logging and saving 
    logdir : str = pathlib.Path.cwd() / 'logs'
    savedir : str = pathlib.Path.cwd() / 'saved_models'
    save_model : bool = True

    
    
    
    # Task hyperparameters 
    dataset : str = 'MNIST'
    epoch : int = 1 # The number of update 
    
    
    # Control hyperparameters
    batch_size : int = 64 
    
    # Habitual network hyperparameters
    temperature : float = 4.0
    lr : float = 0.001
    
    # Loses coefficients
    kl_coeff : float = 5
    # Compression 
    variational_dropout : bool = False
    quantization : bool = False

class DataManager():
    def __init__(self,config):
        self.c = config
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        
    def load_MNIST(self):
        # Loading MNIST
        train_dataset = datasets.MNIST(root='./data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True,
                                    )

        test_dataset = datasets.MNIST(root='./data/',
                                    train=False,
                                    transform=transforms.ToTensor())


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=self.c.batch_size,
                                                shuffle=True,
                                                pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=self.c.batch_size,
                                                shuffle=False,
                                                pin_memory=True)
        return train_loader,test_loader



class Eval:
    def __init__(self, config:Config):
        self.c = config
        self.seed = self.c.seed
        self.device = self.c.device
        self.logger = Logger()
        self.train_data, self.test_data = DataManager(self.c).load_MNIST()
        self.model = CNN_MNIST_Dual(self.c).to(self.device)
        self.model.load_state_dict(torch.load("./saved_models/mnist_cnn.pt"))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Hyperparameters
        self.T = self.c.temperature
        self.kl_coeff = self.c.kl_coeff
    
    def eval(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_data:
            data = data.to(self.device)
            target = target.to(self.device)
    
            z = self.model.encode(data)
            h_prediction = self.model.forward_h(z)
        
            # sum up batch loss
            test_loss += torch.mean(self.criterion(h_prediction, target)).item()
            # Compute accuracy 
            pred = h_prediction.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        test_loss /= len(self.test_data.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_data.dataset),100. * correct / len(self.test_data.dataset)))
        # Return accuracy 
        return correct / len(self.test_data.dataset)




#@hydra.main(version_base=None, config_path="conf", config_name="config")
def main() -> None: # cfg : DictConfig
    # Load default config
    default_config = OmegaConf.structured(Config)
    # Merge default config with run config, run config overrides if there is a conflict
    #config = OmegaConf.merge(default_config, cfg)
    #OmegaConf.save(config, 'config.yaml') 
    config = default_config
    
    eval = Eval(config)
    eval.eval()
    
    
if __name__ == '__main__':
    main()
    print('Finished correctly')


