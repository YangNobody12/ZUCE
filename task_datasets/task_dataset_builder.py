"""
Phase 1: Task Dataset Builder
Constructs balanced, multi-split datasets for Discovery, Validation, and Calibration
satisfying |D_code| ≈ |D_math| ≈ |D_general|.
"""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .prompts_code import CODING_DISCOVERY_PROMPTS, CODING_VALIDATION_PROMPTS, CODING_CALIBRATION_PROMPTS
from .prompts_math import MATH_DISCOVERY_PROMPTS, MATH_VALIDATION_PROMPTS
from .prompts_general import GENERAL_DISCOVERY_PROMPTS, GENERAL_VALIDATION_PROMPTS

class PromptDataset(Dataset):
    """Generic PyTorch Dataset for prompt evaluation and tracing."""
    def __init__(self, prompts: List[str], tokenizer: AutoTokenizer, max_seq_len: int = 512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "raw_text": prompt
        }

class TaskDatasetBuilder:
    """Manages balanced dataset construction for discovery, validation, and calibration."""
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def get_discovery_datasets(self) -> Dict[str, List[str]]:
        """Returns balanced Discovery Set (|D_code| == |D_math| == |D_general|)."""
        min_len = min(len(CODING_DISCOVERY_PROMPTS), len(MATH_DISCOVERY_PROMPTS), len(GENERAL_DISCOVERY_PROMPTS))
        return {
            "coding": CODING_DISCOVERY_PROMPTS[:min_len],
            "math": MATH_DISCOVERY_PROMPTS[:min_len],
            "general": GENERAL_DISCOVERY_PROMPTS[:min_len]
        }

    def get_validation_datasets(self) -> Dict[str, List[str]]:
        """Returns balanced Validation Set for Causal Validation Gates."""
        min_len = min(len(CODING_VALIDATION_PROMPTS), len(MATH_VALIDATION_PROMPTS), len(GENERAL_VALIDATION_PROMPTS))
        return {
            "coding": CODING_VALIDATION_PROMPTS[:min_len],
            "math": MATH_VALIDATION_PROMPTS[:min_len],
            "general": GENERAL_VALIDATION_PROMPTS[:min_len]
        }

    def get_calibration_dataset(self) -> List[str]:
        """Returns high-quality coding instruction-completion pairs for recovery and KD."""
        return CODING_CALIBRATION_PROMPTS

    def get_dataloaders(self, batch_size: int = 2) -> Dict[str, Dict[str, DataLoader]]:
        """Returns PyTorch DataLoaders for discovery and validation splits."""
        disc = self.get_discovery_datasets()
        val = self.get_validation_datasets()

        loaders = {"discovery": {}, "validation": {}}
        for domain, prompts in disc.items():
            ds = PromptDataset(prompts, self.tokenizer, self.max_seq_len)
            loaders["discovery"][domain] = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for domain, prompts in val.items():
            ds = PromptDataset(prompts, self.tokenizer, self.max_seq_len)
            loaders["validation"][domain] = DataLoader(ds, batch_size=batch_size, shuffle=False)

        return loaders
