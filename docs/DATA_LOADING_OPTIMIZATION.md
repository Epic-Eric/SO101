# Data Loading Optimization Guide

This guide explains the data loading optimization features for world model training.

## Overview

The world model training pipeline has been optimized to significantly speed up data loading through:

1. **Parallel Episode Discovery** - Episodes are discovered and validated in parallel
2. **Parallel Episode Loading** - Multiple episodes are loaded concurrently using thread pools
3. **Parallel Image Preprocessing** - When preloading images, they are loaded and transformed in parallel
4. **Smart Worker Configuration** - Automatic optimization based on dataset and storage characteristics

## New Command-Line Options

### Parallel Loading Control

```bash
# Enable/disable parallel loading (enabled by default)
python train_world_model.py data_dir output_dir --no_parallel_loading

# Set number of worker threads for parallel loading (default: 4)
python train_world_model.py data_dir output_dir --loading_workers 8
```

### Configuration File Options

You can also set these in `config.yml`:

```yaml
world_loading_workers: 8  # Number of parallel loading workers
```

## Performance Tuning

### For Local SSD Storage (Best Performance)

```bash
python train_world_model.py data_dir output_dir \
    --num_workers 4 \
    --loading_workers 8 \
    --cache_images \
    --cache_size 4096
```

### For Network/Cloud Storage (e.g., Google Drive)

```bash
python train_world_model.py data_dir output_dir \
    --num_workers 0 \
    --loading_workers 2 \
    --preload_images \
    --preload_dtype float16
```

### For Large Datasets (Memory Constrained)

```bash
python train_world_model.py data_dir output_dir \
    --num_workers 2 \
    --loading_workers 4 \
    --cache_images \
    --cache_size 2048
```

### For Small Datasets (Fits in Memory)

```bash
python train_world_model.py data_dir output_dir \
    --num_workers 0 \
    --loading_workers 4 \
    --preload_images \
    --preload_dtype float16
```

## Performance Impact

### Expected Speedup

With parallel loading enabled:

- **Episode Discovery**: ~2-4x faster with 4 workers
- **Episode Loading**: ~2-3x faster with 4 workers (I/O bound)
- **Image Preprocessing**: ~2-4x faster with parallel preloading

## Backward Compatibility

All optimizations are **backward compatible**:

- Default behavior includes parallel loading
- Old code works without modification
- Can disable parallel loading with `--no_parallel_loading`
- All existing parameters continue to work
