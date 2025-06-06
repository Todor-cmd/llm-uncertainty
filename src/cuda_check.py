import torch

def check_cuda():
    print('CUDA version:', torch.version.cuda)
    print('Built with CUDA:', torch.backends.cuda.is_built())
    print('Is CUDA available:', torch.cuda.is_available())
    print(f"Device count: {torch.cuda.device_count()}")
   

if __name__ == "__main__":
    check_cuda()
