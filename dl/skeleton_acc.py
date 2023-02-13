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
    epoch : int = 1
    
    
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

# Initialize MLP

class CNN_MNIST_Dual(nn.Module):

    def __init__(self,config):
        super(CNN_MNIST_Dual, self).__init__()
        # CNN for MNIST dataset
        self.c = config
        
        self.encoder = nn.Sequential(
                        nn.Conv2d(1, 10, kernel_size=5),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(10, 20, kernel_size=5),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.control = nn.Sequential(
                        nn.Linear(320, 50),
                        nn.ReLU(),
                        nn.Linear(50, 10))
        
        self.habitual = nn.Sequential(
                        nn.Linear(320, 10))
        
        
        """self.entropy_predictor = nn.Sequential(
                                 nn.Linear(320, 50),
                                 nn.ReLU(),
                                 nn.Linear(50, 1))"""
                        

        self.T = self.c.temperature
     
    def forward(self, x):
        z  = self.encoder(x)
        z = z.contiguous().view(-1, 320)
        logit_c = self.control(z)
        logit_h = self.habitual(z)
        
        return logit_c,logit_h
    
    def encode(self, x):
        z  = self.encoder(x)
        z = z.contiguous().view(-1, 320)
        return z
        
    def forward_c(self, z):
        logit_c = self.control(z)
        return logit_c
        
    def forward_h(self, z):
        logit_h = self.habitual(z)
        return logit_h
    
    """def predict_entropy(self, z):
        entropy = self.entropy_predictor(z)
        return entropy"""



class Trainer:
    def __init__(self, config:Config):
        self.c = config
        self.seed = self.c.seed
        self.device = self.c.device
        self.logger = Logger()
        self.train_data, self.test_data = DataManager(self.c).load_MNIST()
        self.model = CNN_MNIST_Dual(self.c).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Hyperparameters
        self.T = self.c.temperature
        self.kl_coeff = self.c.kl_coeff
    def train(self, epoch):
        self.model.train()
        save_loss = []
        save_loss_kl = []
        save_loss_control_hard = []
        save_loss_habitual_hard = []
        save_accuracy = []
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(self.train_data):
                    self.optimizer.zero_grad()
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    # Forward pass
                    z = self.model.encode(data)
                    logit_h = self.model.forward_h(z)
                    logit_c = self.model.forward_c(z)
                    
                    # Compute entropy for all batch
                    #expected_entropy = self.model.predict_entropy(z).squeeze()
                    
                    # Softmax for "real" task
                    softmax_h = F.softmax(logit_h, dim=1)
                    softmax_c = F.softmax(logit_c, dim=1)
                    
                    # Logsoftmax for distillation
                    soft_student = F.log_softmax(logit_h/self.T, dim=1)
                    soft_target =  F.softmax(logit_c/self.T, dim=1)
                    
                    # Compute entropy for all batch 
                    entropy_h = -torch.sum(softmax_h*torch.log(softmax_h),dim=1)
                    entropy_c = -torch.sum(softmax_c*torch.log(softmax_c),dim=1)
                    
    
                    
                    # Compute losses 
                    loss_control_hard = self.criterion(softmax_c, target)
                    loss_habitual_hard = self.criterion(softmax_h, target)/self.T**2
                    loss_kl = F.kl_div(input=soft_student, target=soft_target, reduction='batchmean',log_target=False) 
                    # MSE between entropy and expected entropy
                    #loss_entropy = F.mse_loss(entropy_c, expected_entropy)
                    # Sum up all losses
                    loss = loss_control_hard + loss_habitual_hard + loss_kl # loss_entropy
                    loss.backward()
                    self.optimizer.step() 
                    
                    # Save losses and accuracy       
                    save_loss.append(loss.item()) 
                    save_loss_kl.append(loss_kl.item())
                    save_loss_control_hard.append(loss_control_hard.item())
                    save_loss_habitual_hard.append(loss_habitual_hard.item())
                    
                
                        
                    if batch_idx % 100 == 0:
                        #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e+1, batch_idx * len(data), len(self.train_data.dataset),100. * batch_idx / len(self.train_data), loss.item()))
                        epoch = 'E: {:.0f} '.format(e+1, epoch)
                        loss_line = 'Loss: {:.6f} '.format(save_loss[-1])
                        kl_line = 'KL: {:.6f} '.format(save_loss_kl[-1])
                        control_line = 'C: {:.6f} '.format(save_loss_control_hard[-1])
                        habitual_line = 'H: {:.6f} '.format(save_loss_habitual_hard[-1])
                        percent = '(Completion: {:.0f}%) '.format(100. * batch_idx / len(self.train_data))
                        entropy_co = 'Entropy_C: {:.6f} '.format(entropy_c.mean().item())
                        entropy_ha = 'Entropy_H: {:.6f} '.format(entropy_h.mean().item())     
                        #expected_entropy_p = 'E_entropy: {:.6f} '.format(expected_entropy.mean().item())

                        print(colored(epoch, 'cyan'), end='')
                        print(colored(loss_line, 'red'), end=' ')
                        print(colored(kl_line, 'yellow'), end=' ')
                        print(colored(control_line, 'green'), end=' ')
                        print(colored(habitual_line, 'light_green'), end='')
                        #print(colored(expected_entropy_p , 'light_red'), end='')
                        print(colored(entropy_co, 'dark_grey'), end='')
                        print(colored(entropy_ha, 'light_grey'), end='')
                        print(colored(percent, 'magenta'), end='\n')
                        
                        
                        #compare_beliefs(softmax_c,softmax_h,kl=loss_kl.item(),name1='Control',name2='Habitual',reduction=True)
              
            print('Accuracy at epoch', e+1)
            accuracy  = self.eval().item()
            save_accuracy.append(accuracy)
        # Plot loss and accuracy
        self.logger.add_log('Train loss', save_loss)
        self.logger.add_log('Train loss KL', save_loss_kl)
        self.logger.add_log('Train loss control hard', save_loss_control_hard)
        self.logger.add_log('Train loss habitual hard', save_loss_habitual_hard)
        self.logger.add_log('Accuracy', save_accuracy)
        self.logger.write_to_csv('log.csv')
        # Save model
        if self.c.save_model:
            torch.save(self.model.state_dict(), "./saved_models/mnist_cnn.pt")
    def eval(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_data:
            data = data.to(self.device)
            target = target.to(self.device)
            control_hard,habitual_hard = self.model(data)
        
            # sum up batch loss
            test_loss += torch.mean(self.criterion(control_hard, target)).item()
            # Compute accuracy 
            pred = control_hard.data.max(1, keepdim=True)[1]
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
    
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    """job_num = hydra_cfg.job.num
    print(f'Hydra job number: {job_num}')
    config.job_num = job_num"""
    
    trainer = Trainer(config)
    trainer.train(config.epoch)
    
    
if __name__ == '__main__':
    main()
    print('Finished correctly')