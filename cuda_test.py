import os
import torch

# Try to force CUDA initialization
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("=== CUDA Environment ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Device count: {torch.cuda.device_count()}")

try:
    # Try to initialize CUDA device
    device = torch.device("cuda:0")
    torch.cuda.init()
    print("\n=== CUDA Device Info ===")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    
    # Try to allocate and move tensor to GPU
    x = torch.ones(2, 2)
    x = x.to(device)
    print("\nSuccessfully created tensor on GPU:", x.device)
except Exception as e:
    print(f"\nError during CUDA initialization: {e}") 