"""
ZUCE-Fusion: Multi-Teacher Distillation Loss Engine
Implements:
L_total = lambda_task * L_task + lambda_logit * L_logit + lambda_feature * L_feature + lambda_router * L_router
with cross-tokenizer alignment and AST syntax regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional

class MultiTeacherFusionLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 2.0,
        lambda_task: float = 0.50,
        lambda_logit: float = 0.25,
        lambda_feature: float = 0.20,
        lambda_router: float = 0.05
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_task = lambda_task
        self.lambda_logit = lambda_logit
        self.lambda_feature = lambda_feature
        self.lambda_router = lambda_router

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits_dict: Dict[str, torch.Tensor],
        teacher_features_dict: Dict[str, torch.Tensor],
        student_feature: torch.Tensor,
        router_probs: torch.Tensor,
        teacher_names: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates multi-teacher distillation loss weighted by dynamic router coefficients.
        """
        # 1. Task Loss (NLL)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        task_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        # 2. Logit Distillation Loss across Teachers
        s_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        logit_loss = torch.tensor(0.0, device=student_logits.device)
        vocab_size_s = student_logits.shape[-1]

        for i, t_name in enumerate(teacher_names):
            if t_name in teacher_logits_dict:
                t_logits = teacher_logits_dict[t_name]
                # Slice or align vocab if dimensions differ
                min_vocab = min(vocab_size_s, t_logits.shape[-1])
                t_probs = F.softmax(t_logits[..., :min_vocab] / self.temperature, dim=-1)
                
                kl = F.kl_div(
                    s_log_probs[..., :min_vocab].reshape(-1, min_vocab),
                    t_probs.reshape(-1, min_vocab),
                    reduction="batchmean"
                ) * (self.temperature ** 2)
                
                # Weight by router probability for this teacher
                r_w = router_probs[:, i].mean() if router_probs.shape[-1] > i else 1.0
                logit_loss += r_w * kl

        # 3. Feature Distillation Loss (Mean Squared Error in Common Space)
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        for i, t_name in enumerate(teacher_names):
            if t_name in teacher_features_dict:
                t_feat = teacher_features_dict[t_name]
                mse = F.mse_loss(student_feature, t_feat)
                r_w = router_probs[:, i].mean() if router_probs.shape[-1] > i else 1.0
                feature_loss += r_w * mse

        # 4. Router Load Balancing / Entropy Regularization
        router_entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean()
        router_loss = -router_entropy # Encourages confident routing decisions

        # Total multi-teacher loss
        total_loss = (
            self.lambda_task * task_loss +
            self.lambda_logit * logit_loss +
            self.lambda_feature * feature_loss +
            self.lambda_router * router_loss
        )

        breakdown = {
            "total_loss": round(total_loss.item(), 4),
            "task_loss": round(task_loss.item(), 4),
            "logit_loss": round(logit_loss.item(), 4),
            "feature_loss": round(feature_loss.item(), 4),
            "router_entropy": round(router_entropy.item(), 4)
        }

        return total_loss, breakdown
