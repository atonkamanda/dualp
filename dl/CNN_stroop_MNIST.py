import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import pathlib 
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
import matplotlib.pyplot as plt
from datasets import StroopMNIST
import numpy as np

# Hyperparams   
batch_size = 50
n_epochs = 1
generate_new_data = False
control_cost = 0.025
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0 

stroop_mnist = StroopMNIST(root= './stroop_mnist' , color_mode='fg', train=True, download=True, generate_new_data=False, save_data=False)
stroop_mnist_train = torch.utils.data.Subset(stroop_mnist, range(50000))
stroop_mnist_test = torch.utils.data.Subset(stroop_mnist, range(50000, 60000))
train_loader = torch.utils.data.DataLoader(stroop_mnist_train, batch_size=batch_size, shuffle=True,pin_memory=True) 
test_loader = torch.utils.data.DataLoader(stroop_mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True)

log_dir = pathlib.Path.cwd() / 'logs'
writer = SummaryWriter(log_dir=log_dir)


# Initialize MLP

class FiLMBlock(nn.Module):
    def __init__(self,target_shape):
        super(FiLMBlock, self).__init__()
        
        # Initialize gamma and beta
        self.gamma = nn.Parameter(torch.randn(target_shape[0], target_shape[1], 1, 1))
        self.beta = nn.Parameter(torch.randn(target_shape[0], target_shape[1], 1, 1))
        
      
    def forward(self, x):
        # To note that I could loop and use non linear activations between each modulation 
        x = self.gamma * x + self.beta
        
        # If not the same batch size, then multiply only on the dimensions of the x 

        return x

class CNN_MNIST(nn.Module):

    def __init__(self):
        super(CNN_MNIST, self).__init__()
        # CNN for MNIST dataset
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # Initialize FiLM blocks
        self.film1 = FiLMBlock(target_shape=(50, 10, 24, 24))
        self.film2 = FiLMBlock(target_shape=(50, 20, 8, 8))
        
        self.mode2 = False
    def forward(self, x):
        # (64,3,28,28)
        x = self.conv1(x)
        #if self.mode2==True:
            #x = self.film1(x)
        
        x = F.relu(x)
        # (64,10,24,24)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)

        # (64,20,8,8)   
        if self.mode2==True:
            x = self.film2(x)
        x = F.relu(x)
        #x = self.film2(x)
        x = F.max_pool2d(x, 2)
        x = x.contiguous().view(-1, 320) 
        # (64,320)
        x = self.fc1(x)
        x = F.relu(x)
        # (64,50)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        # Compute entropy of the output distribution
            
        return x 
    


class CNN_critic(nn.Module):    

    def __init__(self):
        super(CNN_critic, self).__init__()
        # CNN for MNIST dataset
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.error_prediction_layer = nn.Linear(10, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # [64, 20, 4, 4] -> [64, 320]
        x = x.contiguous().view(-1, 320) # Use contiguous to avoid error because of view and 
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # Binarry classification head
        x = self.error_prediction_layer(x)
        x = torch.sigmoid(x)
    
        #print(x)
        # Print norm of gradient of the critic
    
     
        return x

        



def train(epoch,model):
    model.train()
    save_loss = []
    save_control_decision = []
    save_accuracy = []
    for e in range(epoch):        
      for batch_idx, (data, target) in enumerate(train_loader):
          optimizer.zero_grad()
          optimizer_critic.zero_grad()
          optimizer_film.zero_grad()
          data = data.to(device)
          # [64, 28, 28, 3] -> [64, 3, 28, 28]
          data = data.permute(0, 3, 1, 2)
          target = target.to(device)
          # Only keep the first element of the (64,2) tensor
          target = target[:,0]
        
          control_decision = torch.mean(critic(data))
          # Print the gradient of the critic
          print(control_decision.grad)
          #print(control_decision)
          if control_decision > 0.5:
                model.mode2 = True
          else:
                model.mode2 = False
          output = model(data)
          loss = criterion(output, target) 
          if control_decision > 0.5:
              loss = loss + loss * control_cost 
          loss.backward()
          print(control_decision.grad)
          optimizer.step()
          optimizer_film.step()
          # Detach the loss from the graph
          #loss = loss.detach()
            
    
          
          #critic_loss = criterion_critic(prediction_error, loss)
          #critic.backward()
          optimizer_critic.step()
          save_loss.append(loss.item())
          save_control_decision.append(control_decision.item())
    
          
                 
          if batch_idx % 100 == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e+1, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
      print('Epoch:', e+1)
      accuracy = test()
      save_accuracy.append(accuracy)
      
      # plot the predicted loss
    plt.plot(save_loss)
    # Save the plot
    plt.savefig('loss.png')
    plt.show()
    
    plt.plot(save_control_decision)
    plt.savefig('control_decision.png')
    plt.show()
    # Merge
    
    plt.plot(save_accuracy)
    plt.savefig('accuracy.png')
    plt.show()
    """ 
    # Merge the three plots
    fig, ax1 = plt.subplots()
    ax1.plot(save_loss, 'b-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(save_prediction_error, 'r-')
    ax2.set_ylabel('Critic Loss', color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.savefig('loss_critic_loss.png')
    plt.show()"""
   
    
            
def test():
    model.eval()
    critic.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        # [64, 28, 28, 3] -> [64, 3, 28, 28]
        data = data.permute(0, 3, 1, 2)
        target = target.to(device)
        target = target[:,0]
        
        # Cognitive control decision 
        control_decision = torch.mean(critic(data))
        if control_decision > 0.5:
                model.mode2 = True
        else:
            model.mode2 = False
        output = model(data)
    
        # sum up batch loss
        test_loss += torch.mean(criterion(output, target)).item()
        # Compute accuracy 
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    # Return the accuracy 
    return correct / len(test_loader.dataset)
        

 



model = CNN_MNIST().to(device)
# Print the number of parameters of the model
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
critic = CNN_critic().to(device)

criterion = nn.CrossEntropyLoss()
criterion_critic = nn.MSELoss()


optimizer_film = optim.Adam(model.film2.parameters(), lr=0.2) # For some reason it doesn't learn with SGD 


optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer_critic = optim.SGD(critic.parameters(), lr=0.01,momentum=0.5)
#optimizer = optim.Adam(critic.parameters(), lr=0.2)
#optimizer_critic = optim.Adam(critic.parameters(), lr=0.2)
train(n_epochs,model)