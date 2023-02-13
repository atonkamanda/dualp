import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt
import torch

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

    # Iterate through each image in the batch
    for i, (image, entropy1, entropy2) in enumerate(zip(images, entropies1, entropies2)):
        # Plot the image
        axs[i].imshow(image.numpy(), cmap='gray')
        # Add the entropies below the image
        axs[i].set_title(f"C: {entropy1.item():.2f} H: {entropy2.item():.2f}")
        # Remove the axis labels
        axs[i].axis('off')

    # Remove the remaining unused axis
    for i in range(num_images, len(axs)):
        fig.delaxes(axs[i])
    
    # Display the plot
    plt.show()



# Create a batch of random images
images = torch.randn(10, 1, 28, 28)

# Compute the entropies of the images
entropies = -torch.sum(torch.exp(images) * images, dim=(1, 2, 3))
entropies2 = -torch.sum(torch.exp(images) * images, dim=(1, 2, 3))

# Display the images with their entropies
display_images_with_entropies(images, entropies,entropies2, num_images_per_row=4)

