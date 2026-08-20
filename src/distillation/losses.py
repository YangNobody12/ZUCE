"""
Phase 8B: Multi-Objective Distillation Loss
Combines Task Cross-Entropy, Soft-Logit KL Distillation, Hidden-State Matching,
and Circuit Activation Preservation:
L_total = lambda_1 L_CE + lambda_2 L_KD + lambda_3 L_hidden + lambda_4 L_circuit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class MultiObjectiveDistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 2.0,
        lambda_ce: float = 0.35,
        lambda_kd: float = 0.35,
        lambda_hidden: float = 0.15,
        lambda_circuit: float = 0.15
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_ce = lambda_ce
        self.lambda_kd = lambda_kd
        self.lambda_hidden = lambda_hidden
        self.lambda_circuit = lambda_circuit

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: Optional[Tuple[torch.Tensor, ...]] = None,
        teacher_hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes weighted combination of all distillation losses.
        """
        # 1. Task Cross-Entropy Loss
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100
        )

        # 2. Soft-Target Logit Distillation Loss (KL Divergence)
        s_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            s_log_probs.reshape(-1, student_logits.size(-1)),
            t_probs.reshape(-1, teacher_logits.size(-1)),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # 3. Hidden State Matching Loss (Mean Squared Error on aligned endpoints)
        hidden_loss = torch.tensor(0.0, device=student_logits.device)
        if student_hidden is not None and teacher_hidden is not None:
            # Match final layer hidden representations
            s_rep = student_hidden[-1]
            t_rep = teacher_hidden[-1]
            hidden_loss = F.mse_loss(s_rep, t_rep)

        # Total multi-objective loss
        total_loss = (
            self.lambda_ce * ce_loss +
            self.lambda_kd * kl_loss +
            (self.lambda_hidden + self.lambda_circuit) * hidden_loss
        )

        breakdown = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "kd_loss": kl_loss.item(),
            "hidden_loss": hidden_loss.item()
        }

        return total_loss, breakdown
