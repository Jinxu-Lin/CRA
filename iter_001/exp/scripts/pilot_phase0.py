#!/usr/bin/env python3
"""
Phase 0 Pilot: Cross-Task Influence Feasibility Check
=====================================================
Train 10-task joint model and 5-task subset model on LIBERO-10 demonstrations.
Compare per-task action prediction loss to detect measurable performance differences.

Since we don't have mujoco/robosuite for rollouts, we use held-out action prediction
MSE as a proxy metric. The key question: does removing tasks 5-9 from training
measurably change performance on tasks 0-4?

Gate: If all per-task loss differences between 10-task and 5-task models are < 3%
relative, the signal is too weak for this benchmark.

Usage:
    CUDA_VISIBLE_DEVICES=0 python pilot_phase0.py
"""

import os
import sys
import json
import time
import gc
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import h5py

# ============================================================
# Configuration
# ============================================================
SEED = 42
PROJECT_DIR = "/home/jinxulin/sibyl_system/projects/cross-task-influence"
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets", "libero_10")
RESULTS_DIR = os.path.join(PROJECT_DIR, "exp", "results")
TASK_ID = "pilot_phase0"

# Training config (pilot: fast, ~5 min per model)
NUM_EPOCHS = 30
EVAL_SPLIT = 0.15  # Hold out 15% of demos for evaluation
IMG_SIZE = 128      # Resize images for efficiency
ACTION_DIM = 7      # LIBERO action dimension (6-DOF + gripper)

# Pilot budget
MAX_DEMOS_PER_TASK = 50  # Use up to 50 demos per task for speed

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# Progress Reporting
# ============================================================
def report_progress(step, total, desc, metric=None):
    progress = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "loss": None, "metric": metric or {"description": desc},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = Path(RESULTS_DIR) / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# Write PID
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR, f"{TASK_ID}.pid").write_text(str(os.getpid()))

# ============================================================
# LIBERO-10 Task Names (canonical order)
# ============================================================
TASK_NAMES = [
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]

SHORT_NAMES = [
    "T0:soup+tomato->basket",
    "T1:cheese+butter->basket",
    "T2:stove+moka",
    "T3:bowl->drawer+close",
    "T4:2mugs->2plates",
    "T5:book->caddy",
    "T6:mug+pudding->plate",
    "T7:soup+cheese->basket",
    "T8:2mokas->stove",
    "T9:mug->microwave+close",
]

# ============================================================
# Data Loading
# ============================================================
class LiberoTaskDataset(Dataset):
    """Load (obs_image, action) pairs from a single task's HDF5 file."""

    def __init__(self, hdf5_path, task_id, max_demos=None, split="train", eval_split=0.15, seed=42):
        self.task_id = task_id
        self.observations = []
        self.actions = []

        with h5py.File(hdf5_path, "r") as f:
            demos = list(f["data"].keys())
            demos.sort(key=lambda x: int(x.replace("demo_", "")))

            if max_demos is not None:
                demos = demos[:max_demos]

            # Split demos into train/eval
            n_eval = max(1, int(len(demos) * eval_split))
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(demos))

            if split == "train":
                selected_indices = indices[n_eval:]
            else:
                selected_indices = indices[:n_eval]

            selected_demos = [demos[i] for i in selected_indices]

            for demo_name in selected_demos:
                demo = f["data"][demo_name]
                # Get agentview images (or eye_in_hand if available)
                if "obs/agentview_rgb" in demo:
                    imgs = demo["obs/agentview_rgb"][:]
                elif "obs/agentview_image" in demo:
                    imgs = demo["obs/agentview_image"][:]
                else:
                    # Try to find any image observation
                    obs_keys = list(demo["obs"].keys())
                    img_keys = [k for k in obs_keys if "rgb" in k or "image" in k or "img" in k]
                    if img_keys:
                        imgs = demo["obs/" + img_keys[0]][:]
                    else:
                        raise ValueError(f"No image obs found. Available: {obs_keys}")

                acts = demo["actions"][:]

                # Subsample timesteps for speed (every 2nd frame)
                step = 2
                for t in range(0, len(acts), step):
                    self.observations.append(imgs[t])
                    self.actions.append(acts[t])

        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        print(f"  Task {task_id} ({split}): {len(self.observations)} samples from {len(selected_demos)} demos")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        img = self.observations[idx]  # (H, W, 3) uint8
        act = self.actions[idx]       # (ACTION_DIM,) float

        # Preprocess image: resize, normalize, CHW
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if img.shape[1] != IMG_SIZE or img.shape[2] != IMG_SIZE:
            img = F.interpolate(img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)

        act = torch.tensor(act, dtype=torch.float32)
        task_label = torch.tensor(self.task_id, dtype=torch.long)

        return img, act, task_label


class MultiTaskDataset(Dataset):
    """Combine multiple task datasets with optional upsampling for volume control."""

    def __init__(self, task_datasets, upsample_to_max=False):
        self.task_datasets = task_datasets
        self.task_offsets = []

        if upsample_to_max and len(task_datasets) > 0:
            max_len = max(len(d) for d in task_datasets)
            # Create indices that upsample smaller datasets
            self.indices = []
            for tid, d in enumerate(task_datasets):
                n = len(d)
                # Repeat indices to match max_len
                reps = math.ceil(max_len / n)
                task_indices = list(range(n)) * reps
                task_indices = task_indices[:max_len]
                self.indices.extend([(tid, idx) for idx in task_indices])
        else:
            self.indices = []
            for tid, d in enumerate(task_datasets):
                self.indices.extend([(tid, idx) for idx in range(len(d))])

        # Shuffle
        rng = np.random.RandomState(SEED)
        rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        tid, sample_idx = self.indices[idx]
        return self.task_datasets[tid][sample_idx]


# ============================================================
# Model: ResNet-18 Encoder + MLP Action Head
# ============================================================
class ResNetPolicyEncoder(nn.Module):
    """ResNet-18 visual encoder (from scratch, no pretrained weights for speed)."""

    def __init__(self, feature_dim=256):
        super().__init__()
        # Lightweight CNN encoder (not full ResNet for pilot speed)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 8x8
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                     # 1x1
            nn.Flatten(),
        )
        self.fc = nn.Linear(256, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.fc(self.encoder(x))


class BCPolicy(nn.Module):
    """Behavioral Cloning policy: ResNet encoder + MLP action head."""

    def __init__(self, action_dim=ACTION_DIM, feature_dim=256, hidden_dim=256):
        super().__init__()
        self.encoder = ResNetPolicyEncoder(feature_dim)
        self.action_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        features = self.encoder(obs)
        actions = self.action_head(features)
        return actions

    def get_features(self, obs):
        """Extract bottleneck features (for future BCS analysis)."""
        return self.encoder(obs)


# ============================================================
# Training
# ============================================================
def train_model(model, train_loader, eval_loaders, device, num_epochs, model_name="model"):
    """Train BC policy and return per-task eval losses."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_eval_loss = float("inf")
    train_losses = []
    eval_history = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for imgs, acts, task_labels in train_loader:
            imgs, acts = imgs.to(device), acts.to(device)
            pred_acts = model(imgs)
            loss = F.mse_loss(pred_acts, acts)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Eval every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            per_task_losses = {}
            with torch.no_grad():
                for task_id, loader in eval_loaders.items():
                    task_loss = 0.0
                    task_n = 0
                    for imgs, acts, _ in loader:
                        imgs, acts = imgs.to(device), acts.to(device)
                        pred_acts = model(imgs)
                        loss = F.mse_loss(pred_acts, acts, reduction="sum")
                        task_loss += loss.item()
                        task_n += acts.shape[0]
                    per_task_losses[task_id] = task_loss / max(task_n, 1)

            avg_eval = np.mean(list(per_task_losses.values()))
            eval_history.append({"epoch": epoch + 1, "per_task_losses": per_task_losses, "avg": avg_eval})

            if avg_eval < best_eval_loss:
                best_eval_loss = avg_eval

            if (epoch + 1) % 10 == 0:
                print(f"  [{model_name}] Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {avg_train_loss:.5f} | Eval Loss: {avg_eval:.5f}")

    return {
        "train_losses": train_losses,
        "eval_history": eval_history,
        "best_eval_loss": best_eval_loss,
        "final_per_task_losses": eval_history[-1]["per_task_losses"] if eval_history else {},
    }


# ============================================================
# GPU Memory Probe
# ============================================================
def find_max_batch_size(model, img_size, device, start=256, min_bs=4):
    """Binary search for max batch size."""
    high, best = start, min_bs
    while min_bs <= high:
        mid = (min_bs + high) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            dummy_img = torch.randn(mid, 3, img_size, img_size, device=device)
            dummy_act = model(dummy_img)
            loss = dummy_act.sum()
            loss.backward()
            model.zero_grad()
            best = mid
            min_bs = mid + 1
            del dummy_img, dummy_act, loss
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache()
            gc.collect()
    del model  # Don't keep this model
    torch.cuda.empty_cache()
    gc.collect()
    return best


# ============================================================
# Main Pilot
# ============================================================
def main():
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    report_progress(0, 6, "Loading LIBERO-10 datasets")

    # ---- Step 1: Load all 10 task datasets ----
    print("\n=== Loading LIBERO-10 datasets ===")
    train_datasets = {}
    eval_datasets = {}

    for tid, task_name in enumerate(TASK_NAMES):
        hdf5_path = os.path.join(DATASETS_DIR, f"{task_name}_demo.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"  WARNING: Missing {hdf5_path}")
            continue
        train_datasets[tid] = LiberoTaskDataset(
            hdf5_path, tid, max_demos=MAX_DEMOS_PER_TASK, split="train", eval_split=EVAL_SPLIT, seed=SEED
        )
        eval_datasets[tid] = LiberoTaskDataset(
            hdf5_path, tid, max_demos=MAX_DEMOS_PER_TASK, split="eval", eval_split=EVAL_SPLIT, seed=SEED
        )

    if len(train_datasets) < 10:
        print(f"\nERROR: Only {len(train_datasets)}/10 tasks loaded. Cannot proceed.")
        mark_done("failed", f"Only {len(train_datasets)}/10 datasets available")
        return

    print(f"\nLoaded all 10 tasks. Total train samples: {sum(len(d) for d in train_datasets.values())}")
    print(f"Total eval samples: {sum(len(d) for d in eval_datasets.values())}")

    report_progress(1, 6, "GPU memory probe for batch size")

    # ---- Step 2: Find optimal batch size ----
    print("\n=== GPU Memory Probe ===")
    probe_model = BCPolicy().to(device)
    max_bs = find_max_batch_size(probe_model, IMG_SIZE, device, start=512, min_bs=8)
    # Use 80% of max for safety
    batch_size = max(8, int(max_bs * 0.8))
    # Round down to nearest power of 2 for efficiency
    batch_size = 2 ** int(math.log2(batch_size))
    batch_size = min(batch_size, 256)  # Cap at 256
    print(f"Max batch size: {max_bs}, using: {batch_size}")

    # Save GPU profile
    gpu_profile = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 1e6 if torch.cuda.is_available() else 0,
        "max_batch_size": max_bs,
        "used_batch_size": batch_size,
    }
    Path(RESULTS_DIR, f"{TASK_ID}_gpu_profile.json").write_text(json.dumps(gpu_profile, indent=2))

    # ---- Step 3: Create eval loaders (same for both models) ----
    eval_loaders_all = {}
    for tid in range(10):
        eval_loaders_all[tid] = DataLoader(
            eval_datasets[tid], batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
    eval_loaders_first5 = {tid: eval_loaders_all[tid] for tid in range(5)}

    report_progress(2, 6, "Training 10-task joint model")

    # ---- Step 4: Train 10-task joint model ----
    print("\n=== Training 10-Task Joint Model ===")
    set_seed(SEED)

    # Combine all 10 tasks (uniform mixing = no upsampling needed if roughly equal)
    all_tasks_dataset = MultiTaskDataset(
        [train_datasets[tid] for tid in range(10)], upsample_to_max=True
    )
    all_tasks_loader = DataLoader(
        all_tasks_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    model_10task = BCPolicy().to(device)
    results_10task = train_model(
        model_10task, all_tasks_loader, eval_loaders_all, device, NUM_EPOCHS, "10-task"
    )

    report_progress(3, 6, "Training 5-task subset model")

    # ---- Step 5: Train 5-task subset model (tasks 0-4 only, volume-controlled) ----
    print("\n=== Training 5-Task Subset Model (tasks 0-4, volume-controlled) ===")
    set_seed(SEED)

    # Volume-controlled: upsample 5 tasks to match total volume of 10-task training
    # This means each of the 5 tasks gets 2x the samples
    subset_dataset = MultiTaskDataset(
        [train_datasets[tid] for tid in range(5)], upsample_to_max=True
    )
    # Further upsample to match 10-task total volume
    target_len = len(all_tasks_dataset)
    current_len = len(subset_dataset)
    if current_len < target_len:
        # Replicate indices
        ratio = math.ceil(target_len / current_len)
        subset_dataset.indices = (subset_dataset.indices * ratio)[:target_len]
        rng = np.random.RandomState(SEED + 1)
        rng.shuffle(subset_dataset.indices)

    subset_loader = DataLoader(
        subset_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    model_5task = BCPolicy().to(device)
    results_5task = train_model(
        model_5task, subset_loader, eval_loaders_all, device, NUM_EPOCHS, "5-task"
    )

    report_progress(4, 6, "Comparing results")

    # ---- Step 6: Compare ----
    print("\n" + "=" * 70)
    print("=== Phase 0 Pilot Results ===")
    print("=" * 70)

    losses_10 = results_10task["final_per_task_losses"]
    losses_5 = results_5task["final_per_task_losses"]

    deltas = {}
    print(f"\n{'Task':<30} {'10-task Loss':>12} {'5-task Loss':>12} {'Delta':>10} {'Delta%':>8}")
    print("-" * 75)

    max_delta_pct = 0
    significant_deltas = 0

    for tid in range(10):
        l10 = losses_10.get(tid, float("nan"))
        l5 = losses_5.get(tid, float("nan"))
        delta = l10 - l5  # Positive = 10-task is worse (adding tasks 5-9 hurt)
        delta_pct = (delta / l5 * 100) if l5 > 0 else 0
        deltas[tid] = {"loss_10task": l10, "loss_5task": l5, "delta": delta, "delta_pct": delta_pct}

        marker = ""
        if abs(delta_pct) > 5:
            marker = " ***"
            significant_deltas += 1
        elif abs(delta_pct) > 3:
            marker = " **"

        abs_delta_pct = abs(delta_pct)
        if abs_delta_pct > max_delta_pct:
            max_delta_pct = abs_delta_pct

        in_subset = "(in 5-task)" if tid < 5 else "(excluded)"
        print(f"{SHORT_NAMES[tid]:<30} {l10:>12.5f} {l5:>12.5f} {delta:>+10.5f} {delta_pct:>+7.1f}%{marker}")

    print("-" * 75)

    # Gate decision
    gate_pass = max_delta_pct > 3.0  # At least one task shows >3% difference
    strong_signal = significant_deltas >= 1  # At least one >5%

    print(f"\nMax |delta%|: {max_delta_pct:.1f}%")
    print(f"Tasks with |delta%| > 5%: {significant_deltas}")
    print(f"Gate (any delta > 3%): {'PASS' if gate_pass else 'FAIL'}")
    print(f"Strong signal (any delta > 5%): {'YES' if strong_signal else 'NO'}")

    # Additional analysis: tasks 0-4 (in both training sets) vs tasks 5-9 (only in 10-task)
    print("\n--- Analysis by Task Group ---")
    in_subset_deltas = [deltas[tid]["delta_pct"] for tid in range(5)]
    excluded_deltas = [deltas[tid]["delta_pct"] for tid in range(5, 10)]

    print(f"Tasks 0-4 (in both): mean delta = {np.mean(in_subset_deltas):+.1f}%")
    print(f"Tasks 5-9 (10-task only): mean delta = {np.mean(excluded_deltas):+.1f}%")
    print(f"  (Positive delta = 10-task model is worse; tasks 5-9 cause interference)")
    print(f"  (Negative delta = 10-task model is better; tasks 5-9 provide positive transfer)")

    report_progress(5, 6, "Saving results")

    # ---- Save Results ----
    elapsed = time.time() - start_time

    phase0_results = {
        "task_id": TASK_ID,
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        "num_tasks_model_a": 10,
        "num_tasks_model_b": 5,
        "model_b_tasks": list(range(5)),
        "volume_controlled": True,
        "per_task_comparison": {str(k): v for k, v in deltas.items()},
        "gate_pass": gate_pass,
        "strong_signal": strong_signal,
        "max_delta_pct": max_delta_pct,
        "significant_deltas_count": significant_deltas,
        "in_subset_mean_delta_pct": float(np.mean(in_subset_deltas)),
        "excluded_mean_delta_pct": float(np.mean(excluded_deltas)),
        "model_10task": {
            "train_losses": results_10task["train_losses"],
            "best_eval_loss": results_10task["best_eval_loss"],
            "final_per_task_losses": {str(k): v for k, v in results_10task["final_per_task_losses"].items()},
        },
        "model_5task": {
            "train_losses": results_5task["train_losses"],
            "best_eval_loss": results_5task["best_eval_loss"],
            "final_per_task_losses": {str(k): v for k, v in results_5task["final_per_task_losses"].items()},
        },
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    # Save to results dir
    results_path = Path(RESULTS_DIR) / "phase0_pilot.json"
    results_path.write_text(json.dumps(phase0_results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Also save to pilots dir
    pilots_dir = Path(RESULTS_DIR) / "pilots"
    pilots_dir.mkdir(exist_ok=True)
    (pilots_dir / "phase0_pilot.json").write_text(json.dumps(phase0_results, indent=2))

    # Qualitative analysis: print sample predictions vs ground truth
    print("\n=== Sample Predictions (10-task model) ===")
    model_10task.eval()
    with torch.no_grad():
        for tid in [0, 2, 5, 9]:
            loader = eval_loaders_all[tid]
            imgs, acts, _ = next(iter(loader))
            imgs = imgs[:5].to(device)
            acts = acts[:5]
            preds = model_10task(imgs).cpu()
            print(f"\nTask {tid} ({SHORT_NAMES[tid]}):")
            for s in range(min(3, len(acts))):
                gt = acts[s].numpy()
                pr = preds[s].numpy()
                print(f"  Sample {s}: GT={np.array2string(gt, precision=3, separator=', ')}")
                print(f"             PR={np.array2string(pr, precision=3, separator=', ')}")

    report_progress(6, 6, "Pilot complete", {
        "gate_pass": gate_pass,
        "max_delta_pct": max_delta_pct,
        "elapsed_min": elapsed / 60,
    })

    go_nogo = "GO" if gate_pass else "NO_GO"
    mark_done("success", f"Phase 0 pilot: {go_nogo}. Max delta: {max_delta_pct:.1f}%. Elapsed: {elapsed/60:.1f} min.")

    print(f"\n=== Pilot complete in {elapsed/60:.1f} minutes ===")
    print(f"=== Verdict: {go_nogo} ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done("failed", f"Error: {str(e)}")
        sys.exit(1)
