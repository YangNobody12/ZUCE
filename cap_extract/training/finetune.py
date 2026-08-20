"""
Phase 7: Domain-Specific Fine-Tuning Engine
Recovers and adapts parameters of the surgically extracted mini model
on domain-specific datasets (Coding / Math / Translation).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..configs.base_config import ExtractionConfig
from ..datasets.prompt_banks import CODING_PROMPTS, MATH_PROMPTS, TRANSLATION_PROMPTS

class TextPromptDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: Any, max_len: int = 512):
        self.encodings = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]

class DomainFineTuner:
    def __init__(self, mini_model: nn.Module, tokenizer: Any, config: Optional[ExtractionConfig] = None):
        self.model = mini_model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()

    def train(
        self,
        domain_prompts: Optional[List[str]] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> nn.Module:
        print("\n" + "="*70)
        print("PHASE 7: DOMAIN-SPECIFIC RECOVERY FINE-TUNING")
        print("="*70)

        epochs = epochs or self.config.ft_epochs
        lr = lr or self.config.ft_learning_rate
        batch_size = batch_size or self.config.ft_batch_size
        save_path = save_dir or os.path.join(self.config.output_dir, "finetuned_mini_model")

        if domain_prompts is None:
            domain = self.config.target_capability
            if domain == "coding":
                domain_prompts = CODING_PROMPTS
            elif domain == "math":
                domain_prompts = MATH_PROMPTS
            else:
                domain_prompts = TRANSLATION_PROMPTS

        dataset = TextPromptDataset(domain_prompts, self.tokenizer, max_len=self.config.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        device = next(self.model.parameters()).device
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps)

        self.model.train()
        print(f"Fine-tuning Mini Model on {len(domain_prompts)} domain samples for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(dataloader))
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # Save fine-tuned weights
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"\n[Phase 7 Complete] Fine-tuned Mini Model saved to: {save_path}")

        return self.model
