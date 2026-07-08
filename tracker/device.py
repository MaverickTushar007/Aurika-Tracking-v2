import os
import logging
import subprocess
import platform
import torch

logger = logging.getLogger("AurikaTracking")

_DEVICE_LOGGED = False

def get_device() -> str:
    """
    Returns the best available PyTorch device in priority order:
    1. CUDA
    2. MPS
    3. CPU
    
    Logs device and hardware details on the first call.
    """
    global _DEVICE_LOGGED
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    if not _DEVICE_LOGGED:
        _DEVICE_LOGGED = True
        logger.info(f"Device : {device.upper()}")
        if device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU    : {gpu_name}")
            except Exception:
                logger.info("GPU    : CUDA Device")
        elif device == "mps":
            gpu_name = "Apple Silicon"
            try:
                if platform.system() == "Darwin":
                    brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                    if brand:
                        gpu_name = brand
            except Exception:
                pass
            logger.info(f"GPU    : {gpu_name}")
            
    return device
