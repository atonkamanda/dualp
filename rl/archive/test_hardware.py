import torch 
import os 
#os.environ['CUDA_VISIBLES_DEVICES'] = '0'


# Test if cuda is available with torch
def test_cuda():
    assert torch.cuda.is_available(), "CUDA is not available"
    # Print the current cuda device
    print(torch.cuda.current_device())
    # Print the name of the current cuda device
    print(torch.cuda.get_device_name(0))
    # Print the total amount of memory available on the current cuda device
    print(torch.cuda.get_device_properties(0).total_memory)
    # Print the number of cuda devices available
    print(torch.cuda.device_count())
    # Print if cuda is enabled for torch
    print(torch.cuda.is_available())
    # Print cuda version
    print(torch.version.cuda)
    # Set pytorch to use cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

test_cuda() # test
