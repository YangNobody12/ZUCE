"""
Phase 8C: Distillation Trainer
Executes Stage A (Calibration) and Stage B (Multi-Objective Knowledge Distillation)
with Cosine Annealing Learning Rate Schedules and Gradient Clipping.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .teacher import TeacherEngine
from .losses import MultiObjectiveDistillationLoss

class TextTrainDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: Any, max_seq_len: int = 512):
        self.items = []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]

class DistillationTrainer:
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        tokenizer: Any,
        device: str = "cuda"
    ):
        self.student = student_model.to(device)
        self.teacher_engine = TeacherEngine(teacher_model.to(device))
        self.tokenizer = tokenizer
        self.device = device

    def run_stage_a_calibration(
        self,
        corpus: List[str],
        epochs: int = 5,
        lr: float = 3e-5,
        batch_size: int = 2
    ) -> float:
        """Stage A: Fast cross-entropy calibration to stabilize residual representations."""
        print(f"\n[Phase 8: Stage A] Residual Stream Calibration ({epochs} epochs, lr={lr})...")
        self.student.train()

        ds = TextTrainDataset(corpus, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=0.01)

        final_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            final_loss = avg_loss
            print(f"  Stage A - Epoch {epoch+1:02d}/{epochs:02d} | Cross-Entropy Loss: {avg_loss:.4f}")

        return final_loss

    def run_stage_b_distillation(
        self,
        corpus: List[str],
        epochs: int = 15,
        lr: float = 5e-5,
        batch_size: int = 2,
        temperature: float = 2.0
    ) -> Dict[str, Any]:
        """Stage B: Full multi-objective knowledge distillation from teacher."""
        print(f"\n[Phase 8: Stage B] Multi-Objective Knowledge Distillation ({epochs} epochs, lr={lr})...")
        self.student.train()

        ds = TextTrainDataset(corpus, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(2, total_steps // 10),
            num_training_steps=total_steps
        )

        loss_fn = MultiObjectiveDistillationLoss(temperature=temperature)

        history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_ce = 0.0
            epoch_kd = 0.0
            epoch_hidden = 0.0

            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                # Teacher pass (no grad)
                t_logits, t_hidden = self.teacher_engine.forward_teacher(input_ids, attention_mask)

                # Student pass
                s_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                s_logits = s_outputs.logits
                s_hidden = s_outputs.hidden_states

                # Shift for next-token prediction
                loss, breakdown = loss_fn(
                    student_logits=s_logits[:, :-1, :],
                    teacher_logits=t_logits[:, :-1, :],
                    labels=labels[:, 1:],
                    student_hidden=s_hidden,
                    teacher_hidden=t_hidden
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += breakdown["total_loss"]
                epoch_ce += breakdown["ce_loss"]
                epoch_kd += breakdown["kd_loss"]
                epoch_hidden += breakdown["hidden_loss"]

            n = len(loader)
            avg_loss = epoch_loss / n
            avg_ce = epoch_ce / n
            avg_kd = epoch_kd / n
            avg_hid = epoch_hidden / n

            history.append(avg_loss)
            print(f"  Stage B - Epoch {epoch+1:02d}/{epochs:02d} | Total: {avg_loss:.4f} | CE: {avg_ce:.4f} | KD: {avg_kd:.4f} | Hidden: {avg_hid:.4f}")

        return {"final_loss": history[-1], "loss_history": history}
