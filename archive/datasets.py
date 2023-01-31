from torchvision.datasets import MNIST
import torch 
import random 
import torchvision 
import numpy as np
import matplotlib.pyplot as plt


class StroopMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    COLORS = ['red', 'green', 'yellow', 'blue', 'orange', 'purple', 'cyan', 'pink', 'greenyellow', 'magenta']
    COLORS_COMB = [('red', 'green'), ('yellow', 'blue'), ('orange', 'purple'), ('cyan', 'purple'), ('greenyellow', 'magenta'),
                   ('red', 'cyan'), ('orange', 'purple'), ('green', 'pink'), ('pink', 'blue'), ('magenta', 'yellow')]
    CHMAP = {
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
 
    def __init__(self, root= './stroop_mnist' , color_mode='fg', train=True, transform=None,
                 transform_fp=None, download=True, generate_new_data=True,save_data=True) -> None:
        super(StroopMNIST, self).__init__(root, train,transform, None, download) 
        self.train = train
        self.transform_fp = transform_fp
        self.color_mode = color_mode

    
        if generate_new_data:
            # Shuffle the data and targets with torch permutation
            perm = torch.randperm(self.data.shape[0])
            self.data = self.data[perm] 
            self.targets = self.targets[perm]
            self.count = 0
            # Split the data into train and test
            # Load existing data if it exists

            self.colored_data = [self.color_fg(self.data[i], self.targets[i]) for i in range(self.data.shape[0])]
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
    

    def color_fg(self, img, target):
        if self.train:
            # Get the congruent color 80% of the time
            if random.random() < 0.8:
                fg_color = StroopMNIST.COLORS[target]
                target = torch.tensor([target,target]) # Congruent sample 
            else: 
                rand_target = random.randint(0, 9) # Get a random incongruent color 
                # make sure the target is not the same as the foreground color 
                while rand_target == target:
                    rand_target = random.randint(0, 9) 

                fg_color = StroopMNIST.COLORS[rand_target]
                target = torch.tensor([target,torch.tensor(rand_target)]) # Incongruent sample
        else:
            rand_target = random.randint(0, 9) # Get a random incongruent color 
                # make sure the target is not the same as the foreground color 
            while rand_target == target:
                 rand_target = random.randint(0, 9) 

            fg_color = StroopMNIST.COLORS[rand_target]
            target = torch.tensor([target,torch.tensor(rand_target)])
        color_img = img.unsqueeze(dim=-1).repeat(1, 1, 3).float() # Repeat the image 3 times to get the 3 channels
        img[img < 75] = 0.0 # Threshold the image to get the foreground
        # img[img >= 75] = 255.0

        
        color_img[img != 0] = StroopMNIST.CHMAP[fg_color] 
        color_img[img == 0] *= torch.tensor([0, 0, 0])
        #color_img = Image.fromarray(color_img.numpy().astype(np.uint8))
        self.count += 1
        if self.count == 50000:
            self.train = False
            
        return color_img/255.0,target

    def color_bg(self, img, target):
        if self.train:
            bg_color = StroopMNIST.COLORS[target]
        else:
            rand_target = random.randint(0, 9)
            bg_color = StroopMNIST.COLORS[rand_target]

        color_img = img.unsqueeze(dim=-1).repeat(1, 1, 3).float()
        img[img < 75] = 0.0
        img[img >= 75] = 255.0

        # color_img /= 255.0
        color_img[img != 0] *= torch.tensor([0, 0, 0])
        color_img[img == 0] = (StroopMNIST.CHMAP[bg_color])
        
        # [64, 28, 28, 3] -> [64, 3, 28, 28]
        #color_img = color_img.permute(0, 3, 1, 2)
       
        
        return color_img

    def color_comb(self, img, target):
        if self.train:
            bg_color, fg_color = StroopMNIST.COLORS_COMB[target]
        else:
            rand_target = random.randint(0, 9)
            bg_color, fg_color = StroopMNIST.COLORS_COMB[rand_target]

        color_img = img.unsqueeze(dim=-1).repeat(1, 1, 3).float()
        img[img < 50] = 0.0

        color_img[img == 0] = StroopMNIST.CHMAP[bg_color]
        color_img[img != 0] = (StroopMNIST.CHMAP[fg_color]) #/255.0)

        return color_img
    




if __name__ == '__main__': # Prevents the code from running when imported as a module   
   
    stroop_mnist = StroopMNIST(root= './stroop_mnist' , color_mode='fg', train=True, download=True, generate_new_data=False, save_data=False)
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

    # Save and display the grid    
    grid = torchvision.utils.make_grid(images)
    grid = grid.numpy()
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid)
    plt.show()
    plt.imsave('./to_delete/train_grid.jpg', grid)

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
    plt.imsave('./to_delete/test_grid.jpg', grid)

    # print labels
    print(' '.join('%5s' % labels[j] for j in range(4)))