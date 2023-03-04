from torchvision.datasets import MNIST
import torch 
import random 
import torchvision 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class StroopMNIST(MNIST): 
    def __init__(self, root= './stroop_mnist', train=True, transform=None,
                 transform_fp=None, download=True, generate_new_data=True,save_data=True) -> None:
        super(StroopMNIST, self).__init__(root, train,transform, None, download) 
        self.train = train
        self.transform_fp = transform_fp
        # Each colors is associated with a particular digit 
        self.colors = ['red', 'green', 'yellow', 'blue', 'orange', 'purple', 'cyan', 'pink', 'greenyellow', 'magenta']
        self.rgb = {
        "red": (torch.tensor([255.0, 0, 0])),
        "green": (torch.tensor([0, 255.0, 0])),
        "yellow": (torch.tensor([255.0, 255.0, 0])),
        "blue": (torch.tensor([0, 0, 255.0])),
        "orange": (torch.tensor([255.0, 165.0, 0])),
        "purple": (torch.tensor([160.0, 32.0, 240.0])),
        "cyan": (torch.tensor([0, 255.0, 255.0])),
        "pink": (torch.tensor([255.0, 192.0, 203.0])),
        "greenyellow": (torch.tensor([173.0, 255.0, 47.0])),
        "magenta": (torch.tensor([255.0, 0, 255.0]))
            }


    
        if generate_new_data:
            # Shuffle the data and targets with torch permutation
            perm = torch.randperm(self.data.shape[0])
            self.data = self.data[perm] 
            self.targets = self.targets[perm]
            self.count = 0
            self.colored_data = [self.color_MNIST(self.data[i], self.targets[i]) for i in range(self.data.shape[0])]
            self.data = [self.colored_data[i][0] for i in range(len(self.data))]
            self.targets = [self.colored_data[i][1] for i in range(len(self.data))]
            self.data = torch.stack(self.data)
            self.targets = torch.stack(self.targets)
            if save_data:
                torch.save(self.data, './stroop_mnist/data.pt')
                torch.save(self.targets, './stroop_mnist/targets.pt')
        
        else:
            # Load existing data if it exists
            self.data = torch.load('./stroop_mnist/data.pt')
            self.targets = torch.load('./stroop_mnist/targets.pt')
    
        
    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        return img, target
    

    def color_MNIST(self, img, target):
        # Get the congruent color 80% of the time
        if self.train:
            if random.random() < 0.8:
                color = self.colors[target]
                target = torch.tensor([target,target]) # Congruent sample 
            else: 
                # Get a random incongruent color 
                rand_target = random.randint(0, 9) 
                while rand_target == target:
                    rand_target = random.randint(0, 9) 
                color = self.colors[rand_target]
                target = torch.tensor([target,torch.tensor(rand_target)]) # Incongruent sample
        else:
            rand_target = random.randint(0, 9) 
            while rand_target == target:
                 rand_target = random.randint(0, 9) 
            color = self.colors[rand_target]
            target = torch.tensor([target,torch.tensor(rand_target)])
        color_img = img.unsqueeze(dim=-1).repeat(1, 1, 3).float() # Repeat the image 3 times to get 3 channels
        img[img < 75] = 0.0 # Threshold the image to get the foreground    
        color_img[img != 0] = self.rgb[color] 


        self.count += 1
        if self.count == 50000:
            self.train = False
            
        return color_img/255.0,target
    




if __name__ == '__main__': # Prevents the code from running when imported as a module   
   
    stroop_mnist = StroopMNIST(root= './stroop_mnist' ,train=True, download=True, generate_new_data=True, save_data=True)
    # Take the 50 0000 first images of stroop mnist train 
    stroop_mnist_train = torch.utils.data.Subset(stroop_mnist, range(50000))
    stroop_mnist_test = torch.utils.data.Subset(stroop_mnist, range(50000, 60000))

    # Load them in dalaloaders
    train_loader = torch.utils.data.DataLoader(stroop_mnist_train, batch_size=64, shuffle=True,pin_memory=True) 
    test_loader = torch.utils.data.DataLoader(stroop_mnist_test, batch_size=64, shuffle=True, pin_memory=True)


    # Display the first images of the dataloader 
    dataiter = iter(train_loader)
    #print(dir(dataiter))
    images, labels = dataiter._next_data()
    # [64, 28, 28, 3] -> [64, 3, 28, 28] for images
    images = images.permute(0, 3, 1, 2)
    # Display and one image per iteration
    for i in range(4):
        plt.imshow(images[i].numpy().transpose((1, 2, 0)))
        plt.show()
        plt.imsave('./to_delete/train_img_{}.png'.format(i), images[i].numpy().transpose((1, 2, 0)))
        # Save the image but upscale it to 28*28
        plt.imsave('./to_delete/train_img_{}_upscaled.png'.format(i), images[i].numpy().transpose((1, 2, 0)), format='png', dpi=1000)


    # Save and display the grid    
    grid = torchvision.utils.make_grid(images)
    grid = grid.numpy()
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid)
    plt.show()
    plt.imsave('./to_delete/train_grid.png', grid)

    # print labels
    print(' '.join('%5s' % labels[j] for j in range(4)))


    # Display the first images of the dataloader 
    dataiter = iter(test_loader)
    images, labels = dataiter._next_data()
    # [64, 28, 28, 3] -> [64, 3, 28, 28] for images
    images = images.permute(0, 3, 1, 2)
    # Save the grid    
    grid = torchvision.utils.make_grid(images)
    grid = grid.numpy()
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid)
    plt.show()
    plt.imsave('./to_delete/test_grid.png', grid)

    # print labels
    print(' '.join('%5s' % labels[j] for j in range(4)))