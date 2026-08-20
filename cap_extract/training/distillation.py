"""
Phase 8: Circuit-Aware Knowledge Distillation Engine
Transfers domain capability and aligns internal circuit representations
from Dense Teacher (1.5B) to Extracted Student (0.5B) using L = L_KD + L_CE + L_Circuit.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .losses import CapabilityDistillationLoss
from .finetune import TextPromptDataset
from ..configs.base_config import ExtractionConfig
from ..datasets.prompt_banks import CODING_PROMPTS, MATH_PROMPTS, TRANSLATION_PROMPTS

class CircuitDistiller:
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: Any,
        config: Optional[ExtractionConfig] = None
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()

        self.loss_fn = CapabilityDistillationLoss(
            alpha_kd=self.config.kd_alpha,
            beta_ce=self.config.ce_beta,
            gamma_circuit=self.config.circuit_gamma,
            temperature=self.config.distill_temperature
        )

    def train_distillation(
        self,
        distill_prompts: Optional[List[str]] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> nn.Module:
        print("\n" + "="*70)
        print("PHASE 8: CIRCUIT-AWARE KNOWLEDGE DISTILLATION (1.5B -> 0.5B)")
        print(f"Loss formulation: {self.config.kd_alpha}*L_KD + {self.config.ce_beta}*L_CE + {self.config.circuit_gamma}*L_Circuit")
        print("="*70)

        epochs = epochs or self.config.ft_epochs
        lr = lr or self.config.ft_learning_rate
        batch_size = batch_size or self.config.ft_batch_size
        save_path = save_dir or os.path.join(self.config.output_dir, "distilled_mini_model")

        if distill_prompts is None:
            domain = self.config.target_capability
            if domain == "coding":
                distill_prompts = CODING_PROMPTS
            elif domain == "math":
                distill_prompts = MATH_PROMPTS
            else:
                distill_prompts = TRANSLATION_PROMPTS

        dataset = TextPromptDataset(distill_prompts, self.tokenizer, max_len=self.config.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        device = next(self.student.parameters()).device
        self.teacher.eval()
        self.student.train()

        optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps)

        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Distillation Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # 1. Forward Teacher (No Grad, output hidden states)
                with torch.no_grad():
                    teacher_out = self.teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    teacher_logits = teacher_out.logits
                    t_hidden = list(teacher_out.hidden_states[1:]) # Drop embedding layer

                # 2. Forward Student (With Grad, output hidden states)
                student_out = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                student_logits = student_out.logits
                s_hidden = list(student_out.hidden_states[1:])

                # 3. Compute Combined Loss
                loss, metrics = self.loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=input_ids,
                    student_hidden=s_hidden,
                    teacher_hidden=t_hidden
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix({
                    "Loss": f"{loss.item():.3f}",
                    "KD": f"{metrics['loss_kd']:.3f}",
                    "Circuit": f"{metrics['loss_circuit']:.3f}"
                })

            avg_loss = epoch_loss / max(1, len(dataloader))
            print(f"Epoch {epoch+1} Average Distillation Loss: {avg_loss:.4f}")

        # Save distilled student model
        os.makedirs(save_path, exist_ok=True)
        self.student.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"\n[Phase 8 Complete] Circuit-Distilled Mini Model saved to: {save_path}")

        return self.student
