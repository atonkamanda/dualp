import torch 
import matplotlib.pyplot as plt


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



def display_images_with_entropies_single(images, entropies, num_images_per_row=5):
    # Unflatten the images to (batch_size, 28, 28)
    images = images.squeeze(1)

    # Get the number of images in the batch
    num_images = images.shape[0]

    # Calculate the number of rows based on the number of images per row
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    # Create a subplot for each image
    fig, axs = plt.subplots(num_rows, num_images_per_row, figsize=(20, 10))

    # Flatten the axis array
    axs = axs.flatten()
    
    # Set the background color of the plot to white
    fig.set_facecolor('white')

    # Iterate through each image in the batch
    for i, (image, entropy) in enumerate(zip(images, entropies)):
        # Plot the image
        axs[i].imshow(image.numpy(), cmap='viridis')
        # Add the entropies below the image
        axs[i].set_title(f"Entropy: {entropy.item():.2f} ", fontsize=12, color='black')
        # Remove the axis labels
        axs[i].axis('off')

    # Remove the remaining unused axis
    for i in range(num_images, len(axs)):
        fig.delaxes(axs[i])
    
    # Display the plot
    plt.show()



def display_images_with_entropies(images, entropies1, entropies2, num_images_per_row=5):
    # Unflatten the images to (batch_size, 28, 28)
    images = images.squeeze(1)

    # Get the number of images in the batch
    num_images = images.shape[0]

    # Calculate the number of rows based on the number of images per row
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    # Create a subplot for each image
    fig, axs = plt.subplots(num_rows, num_images_per_row, figsize=(20, 10))

    # Flatten the axis array
    axs = axs.flatten()

    # Set the background color of the plot to white
    fig.set_facecolor('white')

    # Iterate through each image in the batch
    for i, (image, entropy1, entropy2) in enumerate(zip(images, entropies1, entropies2)):
        # Plot the image
        axs[i].imshow(image.numpy(), cmap='viridis')
        # Add the entropies below the image
        axs[i].set_title(f"Control: {entropy1.item():.2f} Default: {entropy2.item():.2f}", fontsize=12, color='black')
        # Remove the axis labels
        axs[i].axis('off')

    # Remove the remaining unused axis
    for i in range(num_images, len(axs)):
        fig.delaxes(axs[i])
    
    # Display the plot
    plt.savefig('entropies_STROOP.png')
    plt.show()
    
