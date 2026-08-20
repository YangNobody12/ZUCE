"""
Loss functions for Capability Distillation and Knowledge Transfer.
Includes Logit KL Divergence, Task Cross-Entropy, and Intermediate Circuit Activation Matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class CircuitActivationLoss(nn.Module):
    """
    Computes MSE / Cosine distance between Student and Teacher hidden states
    along the retained capability sub-pathways.
    """
    def __init__(self, mode: str = "mse"):
        super().__init__()
        self.mode = mode

    def forward(self, student_hidden: List[torch.Tensor], teacher_hidden: List[torch.Tensor]) -> torch.Tensor:
        if len(student_hidden) == 0 or len(teacher_hidden) == 0:
            return torch.tensor(0.0, device=student_hidden[0].device if student_hidden else "cpu")

        total_loss = 0.0
        num_layers = min(len(student_hidden), len(teacher_hidden))

        for l in range(num_layers):
            s_h = student_hidden[l]
            t_h = teacher_hidden[l]

            if self.mode == "mse":
                total_loss += F.mse_loss(s_h, t_h)
            elif self.mode == "cosine":
                # 1 - Cosine Similarity averaged across tokens
                cos_sim = F.cosine_similarity(s_h, t_h, dim=-1)
                total_loss += (1.0 - cos_sim).mean()

        return total_loss / max(1, num_layers)

class CapabilityDistillationLoss(nn.Module):
    """
    Multi-objective Loss for Capability Extraction:
    L = alpha * L_KD + beta * L_CE + gamma * L_Circuit
    """
    def __init__(
        self,
        alpha_kd: float = 0.4,
        beta_ce: float = 0.3,
        gamma_circuit: float = 0.3,
        temperature: float = 2.0,
        circuit_loss_mode: str = "mse"
    ):
        super().__init__()
        self.alpha = alpha_kd
        self.beta = beta_ce
        self.gamma = gamma_circuit
        self.temperature = temperature
        self.circuit_loss = CircuitActivationLoss(mode=circuit_loss_mode)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: Optional[List[torch.Tensor]] = None,
        teacher_hidden: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Task Cross Entropy Loss
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 2. Logit Knowledge Distillation Loss (KL Divergence with Temperature)
        t = self.temperature
        p_teacher = F.softmax(teacher_logits / t, dim=-1)
        log_p_student = F.log_softmax(student_logits / t, dim=-1)
        loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (t ** 2)

        # 3. Circuit Hidden State Matching Loss
        if student_hidden and teacher_hidden and self.gamma > 0.0:
            loss_circuit = self.circuit_loss(student_hidden, teacher_hidden)
        else:
            loss_circuit = torch.tensor(0.0, device=student_logits.device)

        # Total Weighted Loss
        total_loss = self.alpha * loss_kd + self.beta * loss_ce + self.gamma * loss_circuit

        metrics = {
            "loss_total": total_loss.item(),
            "loss_kd": loss_kd.item(),
            "loss_ce": loss_ce.item(),
            "loss_circuit": loss_circuit.item()
        }

        return total_loss, metrics
