"""
Device Management
=================
Automatic GPU/CPU detection and device configuration.
"""

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """
    Get the best available compute device.
    
    Args:
        preference: Device preference.
            - "auto": Use CUDA if available, else CPU.
            - "cuda": Force CUDA (raises error if unavailable).
            - "cuda:N": Use specific GPU index.
            - "cpu": Force CPU.
            
    Returns:
        torch.device object.
        
    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info(
                f"Using GPU: [bold green]{gpu_name}[/bold green] "
                f"({gpu_memory:.1f} GB)"
            )
        else:
            device = torch.device("cpu")
            logger.info("Using [bold yellow]CPU[/bold yellow] (no CUDA detected)")
    elif preference.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Install CUDA-enabled PyTorch or use device='auto'."
            )
        device = torch.device(preference)
        gpu_name = torch.cuda.get_device_name(device)
        logger.info(f"Using GPU: [bold green]{gpu_name}[/bold green]")
    else:
        device = torch.device("cpu")
        logger.info("Using [bold yellow]CPU[/bold yellow] (forced)")
    
    return device


def get_system_info() -> dict:
    """
    Collect system information for experiment logging.
    
    Returns:
        Dict with Python version, PyTorch version, CUDA info, etc.
    """
    import platform
    import sys
    
    info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": str(torch.backends.cudnn.version()),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
            ),
        })
    else:
        info["cuda_available"] = False
    
    return info
