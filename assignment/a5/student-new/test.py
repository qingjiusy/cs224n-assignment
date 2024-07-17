import torch
print(torch.__version__)  # 查看 PyTorch 版本
print(torch.version.cuda)  # 查看 PyTorch 支持的 CUDA 版本
print(torch.cuda.is_available())