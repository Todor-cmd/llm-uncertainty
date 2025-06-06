import os
import ctypes
import sys

def load_cuda_libs():
    """Try to load CUDA libraries explicitly"""
    cuda_libs = [
        'libcuda.so',
        'libnvidia-ml.so',
        'libcudart.so.12',
        'libnvrtc.so'
    ]
    
    search_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/cuda/lib64',
        '/usr/lib/nvidia',
        os.path.expanduser('~/miniconda/envs/llm-uncertainty/lib')
    ]
    
    loaded_libs = []
    for lib in cuda_libs:
        try:
            # Try to load the library from any of the search paths
            for path in search_paths:
                try:
                    lib_path = os.path.join(path, lib)
                    if os.path.exists(lib_path):
                        ctypes.CDLL(lib_path)
                        loaded_libs.append(lib)
                        print(f"Successfully loaded {lib} from {path}")
                        break
                except Exception as e:
                    continue
        except Exception as e:
            print(f"Failed to load {lib}: {e}")
    
    return loaded_libs

print("=== Attempting to load CUDA libraries ===")
loaded = load_cuda_libs()
print(f"\nSuccessfully loaded libraries: {loaded}")

print("\n=== Now importing PyTorch ===")
import torch

print("\n=== CUDA Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
print(f"Device count: {torch.cuda.device_count()}")

# Try to create a CUDA tensor
try:
    print("\n=== Testing CUDA tensor creation ===")
    if torch.cuda.is_available():
        x = torch.ones(1).cuda()
        print("Successfully created CUDA tensor")
    else:
        print("CUDA not available for tensor creation")
except Exception as e:
    print(f"Error creating CUDA tensor: {e}") 