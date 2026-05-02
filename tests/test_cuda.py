# Script just for testing if CUDA is available and which version of PyTorch and CUDA is installed.

import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)