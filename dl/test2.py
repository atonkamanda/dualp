import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np

def compare_beliefs(list1, list2,name1='Neural net 1',name2='Neural net 2'):
    # Compute the average softmax across all batches
    list1 = torch.mean(list1, dim=0).numpy()
    list2 = torch.mean(list2, dim=0).numpy()

    # Create a bar plot of both lists
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(list1)), list1, color='tab:blue', label=name1)
    ax.bar(np.arange(len(list2)), list2, color='tab:red', label=name2, alpha=0.5)

    # Add labels and title to the plot
    ax.set_xlabel('Classes', fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    ax.set_title('Comparison of Neural Network Beliefs', fontsize=16)
       
    # Set the y-axis limits to [0, 1]
    plt.ylim([0, 1])

    # Overlap between the two lists will be represented by blue bars with red outlines
    overlap = np.minimum(list1, list2)
    """for i in range(len(overlap)):
        if overlap[i] > 0:
            #ax.bar(i, overlap[i], color='lightblue', edgecolor='r', linewidth=3)
            #ax.bar(i, overlap[i], color='b', edgecolor='k', hatch='|', alpha=0.3)
            ax.bar(i, overlap[i], color='g')"""

    #ax.bar(np.arange(len(overlap)), overlap, color='tab:purple', label='Agreement')
    #ax.bar(np.arange(len(overlap)), overlap, color='tab:red', alpha=0.5)
    # Make it transparent and of the color of the lower bar
    diff = list1 - list2
    diff[diff < 0] = 0
    #ax.bar(np.arange(len(diff)), diff, color='g', alpha=0.3)

    ax.legend(fontsize=12)

    # Display the plot
    # Clear plot 
    #plt.clf()
    plt.show()

# Test with two average softmax tensors 
# create two random softmax tensors
list1 = torch.rand(10, 10)
list2 = torch.rand(10, 10)
# Softmax the tensors
list1 = torch.nn.functional.softmax(list1, dim=1)
list2 = torch.nn.functional.softmax(list2, dim=1)
print(list1.shape)
# Compare the two softmax tensors
compare_beliefs(list1, list2)