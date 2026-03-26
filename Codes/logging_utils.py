"""
Logging utilities for CRA experiments.

Supports wandb and tensorboard (selectable via config).
Handles dry-run mode (skip logging initialization).
"""

import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    Unified logging interface for CRA experiments.

    Supports:
    - wandb logging
    - tensorboard logging
    - JSON file logging (always on, as fallback)
    - dry-run mode (skip external logging)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str,
        dry_run: bool = False,
        output_dir: str = "_Data/logs/",
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.dry_run = dry_run
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.start_time = time.time()
        self.log_entries = []

        self._wandb_run = None
        self._tb_writer = None

        repro = config.get("reproducibility", {})
        log_wandb = repro.get("log_wandb", False) and not dry_run
        log_tb = repro.get("log_tensorboard", False) and not dry_run

        # Config hash for identification
        config_str = json.dumps(config, sort_keys=True, default=str)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Git commit hash
        try:
            self.git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()[:8]
        except Exception:
            self.git_hash = "unknown"

        # Initialize wandb
        if log_wandb:
            try:
                import wandb
                wandb_project = repro.get("wandb_project", "CRA")
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config,
                    tags=[f"config:{self.config_hash}", f"git:{self.git_hash}"],
                )
            except Exception as e:
                print(f"[WARNING] wandb init failed: {e}. Falling back to file logging.")

        # Initialize tensorboard
        if log_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.output_dir / "tensorboard" / experiment_name
                self._tb_writer = SummaryWriter(str(tb_dir))
            except Exception as e:
                print(f"[WARNING] tensorboard init failed: {e}.")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics at a given step."""
        if step is not None:
            self.step = step
        else:
            self.step += 1

        entry = {"step": self.step, "wall_time": time.time() - self.start_time}
        entry.update(metrics)
        self.log_entries.append(entry)

        # wandb
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=self.step)
            except Exception:
                pass

        # tensorboard
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, self.step)

    def log_summary(self, summary: Dict[str, Any]):
        """Log summary metrics (e.g., final results)."""
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.summary.update(summary)
            except Exception:
                pass

        # Save to JSON
        summary_path = self.output_dir / f"{self.experiment_name}_summary.json"
        summary["config_hash"] = self.config_hash
        summary["git_hash"] = self.git_hash
        summary["wall_time_seconds"] = time.time() - self.start_time
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def finish(self):
        """Finalize logging."""
        # Save all log entries
        log_path = self.output_dir / f"{self.experiment_name}_log.json"
        with open(log_path, "w") as f:
            json.dump(self.log_entries, f, indent=2)

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        if self._tb_writer is not None:
            self._tb_writer.close()


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for reproducibility logging."""
    info = {"gpu_available": False}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            info["cuda_version"] = torch.version.cuda
            info["pytorch_version"] = torch.__version__
    except Exception:
        pass
    return info
