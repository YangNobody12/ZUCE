"""
HookController: PyTorch Forward/Backward Hook Manager
Manages non-destructive activation, gradient, scaling, and masking hooks across transformer modules.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Tuple, Any

class HookController:
    """Controls hook lifecycles for decoder layers, attention heads, and MLP projections."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.layers = self._detect_layers()

    def _detect_layers(self) -> nn.ModuleList:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        elif hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("Could not locate decoder layers in the model architecture.")

    def num_layers(self) -> int:
        return len(self.layers)

    def register_component_scale_hook(self, layer_idx: int, component_name: str, alpha: float):
        """
        Scales output of a specific component in a specific layer by alpha.
        Components: 'layer', 'self_attn', 'mlp', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
        """
        layer = self.layers[layer_idx]
        target_mod = None

        if component_name == "layer":
            target_mod = layer
        elif component_name == "self_attn" and hasattr(layer, "self_attn"):
            target_mod = layer.self_attn
        elif component_name in ["q_proj", "k_proj", "v_proj", "o_proj"] and hasattr(layer, "self_attn"):
            target_mod = getattr(layer.self_attn, component_name, None)
        elif component_name == "mlp" and hasattr(layer, "mlp"):
            target_mod = layer.mlp
        elif component_name in ["gate_proj", "up_proj", "down_proj"] and hasattr(layer, "mlp"):
            target_mod = getattr(layer.mlp, component_name, None)

        if target_mod is not None:
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (out[0] * alpha, *out[1:])
                return out * alpha
            self.handles.append(target_mod.register_forward_hook(hook))

    def register_neuron_activation_and_gradient(self):
        """Registers capture for SwiGLU intermediate activation and gradient."""
        for l_idx, layer in enumerate(self.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                key = f"mlp_{l_idx}"
                def fwd_hook(k):
                    def h(mod, inp, out):
                        self.activations[k] = inp[0].detach()
                    return h
                def bwd_hook(k):
                    def h(mod, grad_in, grad_out):
                        if grad_in is not None and len(grad_in) > 0 and grad_in[0] is not None:
                            self.gradients[k] = grad_in[0].detach()
                    return h
                self.handles.append(layer.mlp.down_proj.register_forward_hook(fwd_hook(key)))
                self.handles.append(layer.mlp.down_proj.register_full_backward_hook(bwd_hook(key)))

    def register_soft_mask_hook(self, layer_idx: int, mask_tensor: torch.Tensor):
        """Applies continuous soft mask M in [0, 1] to intermediate activations."""
        layer = self.layers[layer_idx]
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            def hook(mod, inp):
                act = inp[0]
                m = mask_tensor.to(device=act.device, dtype=act.dtype)
                return (act * m,)
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))

    def clear(self):
        """Removes all registered handles and cleans activation buffers."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.activations.clear()
        self.gradients.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
