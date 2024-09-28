
def check_gpu_availability(min_memory_gb=4):
    try:
        import torch
    except ImportError:
        return False
    if torch.cuda.is_available():
        device = torch.device('cuda')
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        if total_memory_gb >= min_memory_gb:
            return True
    return False
