# Component: DATE-LM Data Pipeline Wrapper
# Source: research/method-design.md §9 (Probe Code Reuse), Codes/CLAUDE.md
# Ablation config key: evaluation.task, paths.data

"""
DATE-LM data pipeline wrapper.

Wraps DATE-LM's data loading to provide a uniform interface for CRA experiments.
DATE-LM provides: datasets, evaluation protocol, TRAK/Grad-Sim baselines.

This wrapper handles:
1. Loading train/test splits for each task (toxicity, selection, factual)
2. Creating DataLoaders compatible with CRA's representation extraction
3. Interfacing with DATE-LM's evaluation pipeline for LDS/AUPRC computation
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path


class DateLMDataset(Dataset):
    """
    Wrapper around DATE-LM's data format.

    Expects DATE-LM data to be stored as:
    - {data_path}/{task}/train.pt or train/ directory
    - {data_path}/{task}/test.pt or test/ directory

    Each sample is a dict with 'input_ids', 'attention_mask', 'labels'.
    """

    def __init__(
        self,
        data_path: str,
        task: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: Root path to DATE-LM datasets.
            task: One of "toxicity", "selection", "factual".
            split: "train" or "test".
            max_samples: If set, use only first N samples (for debugging/probe).
        """
        self.data_path = Path(data_path)
        self.task = task
        self.split = split

        # Try loading from .pt file first, then from directory
        pt_path = self.data_path / task / f"{split}.pt"
        dir_path = self.data_path / task / split

        if pt_path.exists():
            self.data = torch.load(pt_path, weights_only=False)
        elif dir_path.exists():
            # Load individual files from directory
            self.data = self._load_from_directory(dir_path)
        else:
            raise FileNotFoundError(
                f"DATE-LM data not found at {pt_path} or {dir_path}. "
                f"Please download DATE-LM datasets to {self.data_path}."
            )

        if max_samples is not None:
            self.data = self._truncate(self.data, max_samples)

    def _load_from_directory(self, dir_path: Path) -> Dict[str, torch.Tensor]:
        """Load data from a directory of .pt files."""
        data = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            fpath = dir_path / f"{key}.pt"
            if fpath.exists():
                data[key] = torch.load(fpath, weights_only=False)
        return data

    def _truncate(self, data: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
        """Truncate all tensors to first n samples."""
        return {k: v[:n] for k, v in data.items() if isinstance(v, torch.Tensor)}

    def __len__(self) -> int:
        return len(self.data.get("input_ids", []))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in self.data:
                item[key] = self.data[key][idx]
        return item


def create_dataloader(
    data_path: str,
    task: str,
    split: str = "train",
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for DATE-LM data.

    Args:
        data_path: Root path to DATE-LM datasets.
        task: "toxicity", "selection", or "factual".
        split: "train" or "test".
        batch_size: Batch size.
        max_samples: Maximum number of samples.
        num_workers: Number of data loading workers.

    Returns:
        DataLoader yielding dicts with 'input_ids', 'attention_mask', 'labels'.
    """
    dataset = DateLMDataset(data_path, task, split, max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Attribution requires fixed ordering
        num_workers=num_workers,
        pin_memory=True,
    )


def get_task_labels(
    data_path: str,
    task: str,
    split: str = "train",
) -> Optional[torch.Tensor]:
    """
    Get binary labels for retrieval metrics (AUPRC, P@K, Recall, MRR).

    For toxicity: labels indicate unsafe/safe samples.
    For factual: labels indicate relevant/irrelevant samples for each test query.
    For selection: no binary labels (LDS only).

    Args:
        data_path: Root path to DATE-LM datasets.
        task: "toxicity", "selection", or "factual".
        split: "train" or "test".

    Returns:
        Binary labels tensor, or None if task has no binary labels.
    """
    label_path = Path(data_path) / task / f"{split}_labels.pt"
    if label_path.exists():
        return torch.load(label_path, weights_only=False)
    return None


def get_actual_changes(
    data_path: str,
    task: str,
) -> Optional[torch.Tensor]:
    """
    Get actual model output changes for LDS computation.

    DATE-LM provides pre-computed LOO (leave-one-out) or subset removal effects.

    Args:
        data_path: Root path to DATE-LM datasets.
        task: "toxicity", "selection", or "factual".

    Returns:
        (N,) tensor of actual changes, or None if not available.
    """
    changes_path = Path(data_path) / task / "actual_changes.pt"
    if changes_path.exists():
        return torch.load(changes_path, weights_only=False)
    return None


def available_tasks(data_path: str) -> List[str]:
    """
    List available DATE-LM tasks at the given path.

    Args:
        data_path: Root path to DATE-LM datasets.

    Returns:
        List of task names.
    """
    root = Path(data_path)
    if not root.exists():
        return []
    return [d.name for d in root.iterdir() if d.is_dir() and d.name in {"toxicity", "selection", "factual"}]
