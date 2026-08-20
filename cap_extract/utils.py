"""
Common utility functions for device placement and tokenization handling.
"""

import torch
from typing import Dict, Any, Union

def prepare_inputs(inputs: Any, device: Union[str, torch.device]) -> Dict[str, torch.Tensor]:
    """
    Ensures input tensors from tokenizer (whether BatchEncoding or dict)
    are placed on the correct target device.
    """
    if hasattr(inputs, "to"):
        return inputs.to(device)
    elif isinstance(inputs, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    else:
        return inputs
