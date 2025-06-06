import torch

def check_cuda():
    """
    Check if CUDA is available and print device information
    """
    print("=== CUDA Availability ===")
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"✓ CUDA is available. Using device: {device}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ PyTorch version: {torch.__version__}")
        cuda_available = True
    else:
        print("❌ CUDA is not available. Using CPU")
        cuda_available = False
    
    print("\n=== GPTQ Extensions ===")
    
    # Check auto-gptq
    try:
        import auto_gptq
        print(f"✓ auto-gptq installed: {auto_gptq.__version__}")
        
        # Check if CUDA kernels are available
        try:
            from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
            QuantLinear = dynamically_import_QuantLinear(use_triton=False, desc_act=False, group_size=-1, bits=4, disable_exllama=True)
            print("✓ auto-gptq CUDA kernels available")
        except Exception as e:
            print(f"❌ auto-gptq CUDA kernels not available: {e}")
            
    except ImportError:
        print("❌ auto-gptq not installed")

    
    print("\n=== Summary ===")
    if cuda_available:
        print("✓ CUDA is ready for GPU acceleration")
        
        # Test actual CUDA functionality
        try:
            print("\n=== CUDA Functionality Test ===")
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = test_tensor * 2
            print(f"✓ CUDA tensor operations working: {result.cpu().tolist()}")
        except Exception as e:
            print(f"❌ CUDA tensor operations failed: {e}")
            cuda_available = False
    else:
        print("❌ CUDA not available - will use CPU (slow)")
        
    return cuda_available

if __name__ == "__main__":
    check_cuda()
