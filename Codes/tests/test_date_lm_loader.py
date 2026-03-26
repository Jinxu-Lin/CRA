"""Tests for core/data/date_lm_loader.py"""

import pytest
import torch
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.data.date_lm_loader import (
    DateLMDataset, create_dataloader, get_task_labels, available_tasks,
)


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock DATE-LM data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = os.path.join(tmpdir, "toxicity")
        os.makedirs(task_dir)

        n_samples = 20
        seq_len = 32

        # Create mock train.pt
        train_data = {
            "input_ids": torch.randint(0, 1000, (n_samples, seq_len)),
            "attention_mask": torch.ones(n_samples, seq_len).long(),
            "labels": torch.randint(0, 1000, (n_samples, seq_len)),
        }
        torch.save(train_data, os.path.join(task_dir, "train.pt"))

        # Create mock test.pt
        test_data = {
            "input_ids": torch.randint(0, 1000, (5, seq_len)),
            "attention_mask": torch.ones(5, seq_len).long(),
            "labels": torch.randint(0, 1000, (5, seq_len)),
        }
        torch.save(test_data, os.path.join(task_dir, "test.pt"))

        # Create mock labels
        labels = torch.randint(0, 2, (n_samples,))
        torch.save(labels, os.path.join(task_dir, "train_labels.pt"))

        yield tmpdir


class TestForwardShape:
    """Verify data loading produces correct shapes."""

    def test_dataset_length(self, mock_data_dir):
        ds = DateLMDataset(mock_data_dir, "toxicity", "train")
        assert len(ds) == 20

    def test_dataset_item_keys(self, mock_data_dir):
        ds = DateLMDataset(mock_data_dir, "toxicity", "train")
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_dataset_item_shape(self, mock_data_dir):
        ds = DateLMDataset(mock_data_dir, "toxicity", "train")
        item = ds[0]
        assert item["input_ids"].shape == (32,)

    def test_dataloader_batch_shape(self, mock_data_dir):
        dl = create_dataloader(mock_data_dir, "toxicity", "train", batch_size=4)
        batch = next(iter(dl))
        assert batch["input_ids"].shape == (4, 32)

    def test_max_samples(self, mock_data_dir):
        ds = DateLMDataset(mock_data_dir, "toxicity", "train", max_samples=5)
        assert len(ds) == 5

    def test_test_split(self, mock_data_dir):
        ds = DateLMDataset(mock_data_dir, "toxicity", "test")
        assert len(ds) == 5


class TestGradientFlow:
    """Data loading is not differentiable; verify loaded tensors are usable."""

    def test_loaded_tensors_require_no_grad(self, mock_data_dir):
        ds = DateLMDataset(mock_data_dir, "toxicity", "train")
        item = ds[0]
        assert not item["input_ids"].requires_grad


class TestOutputRange:
    """Verify loaded data is valid."""

    def test_no_nan_inf(self, mock_data_dir):
        ds = DateLMDataset(mock_data_dir, "toxicity", "train")
        item = ds[0]
        assert torch.isfinite(item["input_ids"].float()).all()

    def test_labels_shape(self, mock_data_dir):
        labels = get_task_labels(mock_data_dir, "toxicity", "train")
        assert labels is not None
        assert labels.shape == (20,)

    def test_labels_binary(self, mock_data_dir):
        labels = get_task_labels(mock_data_dir, "toxicity", "train")
        assert set(labels.unique().tolist()).issubset({0, 1})


class TestConfigSwitch:
    """Test task/split switching."""

    def test_available_tasks(self, mock_data_dir):
        tasks = available_tasks(mock_data_dir)
        assert "toxicity" in tasks

    def test_nonexistent_task_raises(self, mock_data_dir):
        with pytest.raises(FileNotFoundError):
            DateLMDataset(mock_data_dir, "nonexistent_task", "train")

    def test_missing_labels_returns_none(self, mock_data_dir):
        labels = get_task_labels(mock_data_dir, "toxicity", "test")
        # test_labels.pt was not created
        assert labels is None

    def test_nonexistent_path(self):
        tasks = available_tasks("/nonexistent/path/123456")
        assert tasks == []
