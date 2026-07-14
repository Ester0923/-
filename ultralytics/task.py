import torch
print(torch.__version__)         # PyTorch 版本
print(torch.version.cuda)        # CUDA 版本
print(torch.cuda.is_available()) # 是否能使用 GPU
print(torch.cuda.get_device_name(0)) # GPU 名称
