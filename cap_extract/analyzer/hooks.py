"""
PyTorch Hook Management System for Capability-aware Model Analysis.
Provides clean forward/backward hook registration, activation/gradient capturing,
and dynamic neuron/layer scaling and masking.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Any, Tuple

class HookManager:
    """
    Context manager and controller for registering forward/backward hooks
    on Transformers (Qwen, Llama, Mistral, etc.) architectures.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.head_activations: Dict[str, torch.Tensor] = {}
        self.custom_hooks: Dict[str, Callable] = {}
        
        # Identify layers in HuggingFace Transformer architecture
        self.layers = self._get_layers()

    def _get_layers(self) -> nn.ModuleList:
        """Find the ModuleList of decoder layers in the model."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        elif hasattr(self.model, "layers"):
            return self.model.layers
        else:
            raise AttributeError("Unable to automatically detect decoder layers in the model.")

    def num_layers(self) -> int:
        return len(self.layers)

    def register_activation_capture(self, target_modules: str = "mlp_intermediate"):
        """
        Register hooks to capture activations during forward pass.
        target_modules: 'layer', 'mlp', 'mlp_intermediate', 'attn', 'all'
        """
        for l_idx, layer in enumerate(self.layers):
            if target_modules in ["layer", "all"]:
                name = f"layer_{l_idx}"
                def get_hook(key):
                    def hook(mod, inp, out):
                        val = out[0] if isinstance(out, tuple) else out
                        self.activations[key] = val.detach()
                    return hook
                self.handles.append(layer.register_forward_hook(get_hook(name)))

            if target_modules in ["mlp", "all"]:
                name = f"mlp_{l_idx}"
                if hasattr(layer, "mlp"):
                    def get_hook(key):
                        def hook(mod, inp, out):
                            val = out[0] if isinstance(out, tuple) else out
                            self.activations[key] = val.detach()
                        return hook
                    self.handles.append(layer.mlp.register_forward_hook(get_hook(name)))

            if target_modules in ["mlp_intermediate", "all"]:
                # Captures the pre-down_proj activation (after SwiGLU: silu(gate) * up)
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                    name = f"mlp_neuron_{l_idx}"
                    def get_hook(key):
                        def hook(mod, inp, out):
                            # inp[0] is the intermediate activation fed into down_proj
                            self.activations[key] = inp[0].detach()
                        return hook
                    self.handles.append(layer.mlp.down_proj.register_forward_hook(get_hook(name)))

            if target_modules in ["attn", "all"]:
                name = f"attn_{l_idx}"
                if hasattr(layer, "self_attn"):
                    def get_hook(key):
                        def hook(mod, inp, out):
                            val = out[0] if isinstance(out, tuple) else out
                            self.activations[key] = val.detach()
                        return hook
                    self.handles.append(layer.self_attn.register_forward_hook(get_hook(name)))

    def register_gradient_capture(self, target_modules: str = "mlp_intermediate"):
        """
        Register backward hooks to capture gradients during backward pass.
        """
        for l_idx, layer in enumerate(self.layers):
            if target_modules in ["mlp_intermediate", "all"]:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                    name = f"mlp_neuron_{l_idx}"
                    def get_hook(key):
                        def hook(mod, grad_in, grad_out):
                            # grad_in[0] is gradient w.r.t intermediate input to down_proj
                            if grad_in is not None and len(grad_in) > 0 and grad_in[0] is not None:
                                self.gradients[key] = grad_in[0].detach()
                        return hook
                    self.handles.append(layer.mlp.down_proj.register_full_backward_hook(get_hook(name)))

            if target_modules in ["layer", "all"]:
                name = f"layer_{l_idx}"
                def get_hook(key):
                    def hook(mod, grad_in, grad_out):
                        if grad_out is not None and len(grad_out) > 0 and grad_out[0] is not None:
                            self.gradients[key] = grad_out[0].detach()
                    return hook
                self.handles.append(layer.register_full_backward_hook(get_hook(name)))

    def register_layer_scale_hook(self, layer_idx: int, alpha: float):
        """Scale the output of a specific layer by alpha."""
        layer = self.layers[layer_idx]
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] * alpha, *out[1:])
            return out * alpha
        self.handles.append(layer.register_forward_hook(hook))

    def register_mlp_scale_hook(self, layer_idx: int, alpha: float):
        """Scale the output of a specific MLP by alpha."""
        if hasattr(self.layers[layer_idx], "mlp"):
            mlp = self.layers[layer_idx].mlp
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (out[0] * alpha, *out[1:])
                return out * alpha
            self.handles.append(mlp.register_forward_hook(hook))

    def register_attn_scale_hook(self, layer_idx: int, alpha: float):
        """Scale the output of a specific Attention module by alpha."""
        if hasattr(self.layers[layer_idx], "self_attn"):
            attn = self.layers[layer_idx].self_attn
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (out[0] * alpha, *out[1:])
                return out * alpha
            self.handles.append(attn.register_forward_hook(hook))

    def register_neuron_mask_hook(self, layer_idx: int, mask: torch.Tensor):
        """
        Apply a binary or continuous neuron mask M (1D tensor of size intermediate_size)
        to the intermediate MLP activations before down_proj.
        """
        if hasattr(self.layers[layer_idx], "mlp") and hasattr(self.layers[layer_idx].mlp, "down_proj"):
            down_proj = self.layers[layer_idx].mlp.down_proj
            def hook(mod, inp):
                # inp is tuple (intermediate_activation, )
                act = inp[0]
                m = mask.to(act.device, dtype=act.dtype)
                # Broadcast mask across batch and seq_len
                masked_act = act * m
                return (masked_act,)
            self.handles.append(down_proj.register_forward_pre_hook(hook))

    def clear(self):
        """Remove all registered hooks and clear stored buffers."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()
        self.gradients.clear()
        self.head_activations.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
