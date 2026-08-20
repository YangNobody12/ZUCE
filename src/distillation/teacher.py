"""
Phase 8A: Teacher Model Engine
Provides frozen teacher forward passes, logits, and intermediate hidden states.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

class TeacherEngine:
    def __init__(self, teacher_model: nn.Module):
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward_teacher(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Computes teacher logits and intermediate hidden states with zero gradient overhead.
        """
        with torch.no_grad():
            outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            return outputs.logits, outputs.hidden_states
