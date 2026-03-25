# Environment Setup Notes

## Key Decision: Env on /mnt/hdd_1

The conda environment is located at `/mnt/hdd_1/share/jinxulin_envs/sibyl_cross-task-influence`
(NOT in `~/miniconda3/envs/`) because the home directory is over quota (108G/100G limit).

### Correct run command:
```bash
/home/jinxulin/miniconda3/bin/conda run --no-capture-output -p /mnt/hdd_1/share/jinxulin_envs/sibyl_cross-task-influence [command]
```

### For pip installs (must bypass home quota):
```bash
export TMPDIR=/mnt/hdd_1/share/jinxulin_envs/tmp
export PIP_CACHE_DIR=/mnt/hdd_1/share/jinxulin_envs/pip_cache
```

## LIBERO Installation
- LIBERO is installed via `.pth` file pointing to `/home/jinxulin/sibyl_system/projects/cross-task-influence/libero_repo`
- The `.git` directory was removed to save space (300MB)
- LIBERO config at `~/.libero/config.yaml` with datasets at `/home/jinxulin/sibyl_system/projects/cross-task-influence/datasets`
- LIBERO-10 datasets NOT yet downloaded (need to download demonstration HDF5 files)

## TECA Env Repair
- During setup, accidentally removed torch/scipy/matplotlib/sklearn from sibyl_TECA env
- Fixed by: torch installed to `/mnt/hdd_1/share/jinxulin_envs/shared_torch` and added via `.pth` file
- scipy/matplotlib/sklearn reinstalled directly to TECA env

## Disk Quota Warning
- Home quota: 100GB, currently at 107GB (over quota, 6-day grace)
- All new large files MUST go to `/mnt/hdd_1/share/jinxulin_envs/` or similar external storage
- Model checkpoints and datasets should use `/mnt/hdd_1/` paths
