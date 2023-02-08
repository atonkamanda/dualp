import pandas as pd
import cv2 
from torch.utils.data import IterableDataset, DataLoader
import random
import torch
import torch.nn as nn
import numpy as np
# Import matplotlib
import matplotlib.pyplot as plt



class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()
        
        self.dim = dim
        self.max_alpha = alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)
        
    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        
        alpha = self.log_alpha.exp()
        
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha**2 + c3 * alpha**3
        
        kl = -negative_kl
        
        return kl.mean()
    
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = torch.randn(x.size())
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x



class Logger:
    def __init__(self):
        self.log = {}
    def add_log(self,feature_name,value):
        self.log[feature_name] = value       
    def fill_missing_values(self,data: dict) -> dict:
        max_len = max([len(v) for v in data.values()])
        for key in data:
            if len(data[key]) < max_len:
                data[key] += [None] * (max_len - len(data[key]))
        return data
    def write_to_csv(self, file_name):
        filled_data = self.fill_missing_values(self.log)
        df = pd.DataFrame(filled_data)
        df.to_csv(file_name, index=False)
        
    def write_video(self,filepath,frames, fps=60):
        """ Write a video to disk using openCV
            filepath : the path to write the video to 
            frames : a numpy array with shape (time, height, width, channels)
            
        
        """
        height, width, channels = frames.shape[1:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()


def compare_beliefs(softmax1, softmax2, kl, name1='Softmax1', name2="Softmax2",reduction=False):
    if reduction == True:
        softmax1 = torch.mean(softmax1, dim=0).detach().numpy()
        softmax2 = torch.mean(softmax2, dim=0).detach().numpy()

    categories = range(len(softmax1))
    colors = ['red', 'blue', 'green', 'purple', 'yellow', 'pink', 'brown', 'orange', 'gray', 'black']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 2], 'wspace': 0.3})
    y_max = max(max(softmax1), max(softmax2))
    for i, softmax in enumerate([softmax1, softmax2]):
        ax = [ax1, ax2][i]
        for j, b in enumerate(softmax):
            ax.bar(j, b, color=colors[j], width=0.8, edgecolor='black')
        ax.set_xlim(-1, len(softmax))
        ax.set_ylim(0, y_max)
        ax.set_xticks(categories)
        ax.set_xlabel('Categories', fontsize=12)
        if i == 0:
            ax.set_ylabel('Probability', fontsize=12)
        ax.set_title([name1, name2][i], fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
    fig.text(0.5, 0.9, 'KL: {:.2f}'.format(kl), ha='center', fontsize=12, fontweight='bold')
    #plt.tight_layout()
    plt.show()


def set_seed(seed : int,device):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)