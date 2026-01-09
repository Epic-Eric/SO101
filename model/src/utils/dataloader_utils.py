"""
Utility functions for optimizing data loading performance in world model training.

This module provides:
- Performance profiling utilities for data loading
- Memory-efficient data structure recommendations
- Optimization strategies for different storage backends
"""

import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DataLoadingProfile:
    """Profile results from data loading performance measurement."""
    total_time: float
    episodes_loaded: int
    images_loaded: int
    avg_episode_time: float
    avg_image_time: float
    peak_memory_mb: Optional[float] = None
    
    def __str__(self):
        return (
            f"DataLoadingProfile(\n"
            f"  total_time={self.total_time:.2f}s\n"
            f"  episodes_loaded={self.episodes_loaded}\n"
            f"  images_loaded={self.images_loaded}\n"
            f"  avg_episode_time={self.avg_episode_time*1000:.2f}ms\n"
            f"  avg_image_time={self.avg_image_time*1000:.2f}ms\n"
            f"  peak_memory_mb={self.peak_memory_mb:.2f} MB\n"
            f")"
        )


def get_optimal_workers(storage_type: str = "local", num_episodes: int = 1) -> Tuple[int, int]:
    """
    Get optimal worker configuration for data loading.
    
    Args:
        storage_type: Type of storage ("local", "network", "cloud")
        num_episodes: Number of episodes in the dataset
    
    Returns:
        Tuple of (num_workers, loading_workers) for DataLoader and parallel loading
    """
    if storage_type == "network" or storage_type == "cloud":
        # For network/cloud storage, use fewer workers to avoid overwhelming the network
        num_workers = 0  # Single-threaded DataLoader
        loading_workers = 2  # Minimal parallel loading
    elif storage_type == "local":
        # For local storage, use more workers
        if num_episodes < 5:
            num_workers = 2
            loading_workers = 2
        elif num_episodes < 20:
            num_workers = 4
            loading_workers = 4
        else:
            num_workers = 4
            loading_workers = 8
    else:
        # Default conservative settings
        num_workers = 2
        loading_workers = 4
    
    return num_workers, loading_workers


def get_cache_recommendations(
    num_images: int,
    image_size: int = 64,
    available_memory_gb: Optional[float] = None,
) -> Dict[str, any]:
    """
    Get cache configuration recommendations based on dataset size and available memory.
    
    Args:
        num_images: Total number of images in dataset
        image_size: Size of images (assumed square)
        available_memory_gb: Available system memory in GB (auto-detect if None)
    
    Returns:
        Dictionary with cache recommendations
    """
    # Estimate memory per image (RGB, float32)
    bytes_per_image = image_size * image_size * 3 * 4  # 4 bytes per float32
    total_dataset_mb = (num_images * bytes_per_image) / (1024 * 1024)
    
    # Auto-detect available memory
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            available_memory_gb = 8.0  # Conservative default
    
    available_memory_mb = available_memory_gb * 1024
    
    # Decision logic
    if total_dataset_mb < available_memory_mb * 0.5:
        # Dataset fits comfortably in memory
        return {
            "preload_images": True,
            "preload_dtype": "float16",  # Use float16 to save memory
            "cache_images": False,
            "cache_size": 0,
            "num_workers": 0,  # No need for workers with preloaded data
            "reason": f"Dataset ({total_dataset_mb:.0f} MB) fits in available memory ({available_memory_mb:.0f} MB)",
        }
    elif total_dataset_mb < available_memory_mb * 0.8:
        # Dataset fits but might be tight
        return {
            "preload_images": True,
            "preload_dtype": "float16",
            "cache_images": False,
            "cache_size": 0,
            "num_workers": 0,
            "reason": f"Dataset ({total_dataset_mb:.0f} MB) fits with caution in available memory ({available_memory_mb:.0f} MB)",
        }
    else:
        # Dataset too large for full preload, use caching
        cache_size = int(available_memory_mb * 0.3 / (bytes_per_image / (1024 * 1024)))
        return {
            "preload_images": False,
            "preload_dtype": "float16",
            "cache_images": True,
            "cache_size": cache_size,
            "num_workers": 2,
            "reason": f"Dataset ({total_dataset_mb:.0f} MB) too large for preload, using LRU cache ({cache_size} images)",
        }


def print_optimization_tips(
    num_episodes: int,
    num_images: int,
    storage_type: str = "local",
    available_memory_gb: Optional[float] = None,
):
    """
    Print optimization tips for data loading based on dataset characteristics.
    
    Args:
        num_episodes: Number of episodes
        num_images: Total number of images
        storage_type: Type of storage backend
        available_memory_gb: Available memory in GB
    """
    print("\n" + "="*60)
    print("Data Loading Optimization Tips")
    print("="*60)
    
    # Worker recommendations
    num_workers, loading_workers = get_optimal_workers(storage_type, num_episodes)
    print(f"\n📊 Dataset: {num_episodes} episodes, {num_images} images")
    print(f"💾 Storage: {storage_type}")
    
    print(f"\n⚙️  Recommended Worker Configuration:")
    print(f"  --num_workers {num_workers}")
    print(f"  --loading_workers {loading_workers}")
    
    # Cache recommendations
    cache_config = get_cache_recommendations(num_images, available_memory_gb=available_memory_gb)
    print(f"\n🗂️  Recommended Cache Configuration:")
    for key, value in cache_config.items():
        if key == "reason":
            print(f"  Reason: {value}")
        elif isinstance(value, bool):
            if value:
                print(f"  --{key}")
        elif isinstance(value, (int, str)) and key != "reason":
            print(f"  --{key} {value}")
    
    # Storage-specific tips
    print(f"\n💡 Storage-Specific Tips:")
    if storage_type == "network" or storage_type == "cloud":
        print("  - Consider copying data to local storage first")
        print("  - Use --preload_images to load all data once")
        print("  - Reduce --loading_workers to avoid overwhelming network")
    elif storage_type == "local":
        print("  - SSD storage provides best performance")
        print("  - Use parallel loading with multiple workers")
        print("  - Consider --cache_images for frequently accessed data")
    
    print("\n" + "="*60 + "\n")


def estimate_loading_time(
    num_episodes: int,
    images_per_episode: int,
    parallel_loading: bool = True,
    loading_workers: int = 4,
    avg_image_load_time_ms: float = 5.0,
) -> float:
    """
    Estimate total data loading time.
    
    Args:
        num_episodes: Number of episodes
        images_per_episode: Average images per episode
        parallel_loading: Whether parallel loading is enabled
        loading_workers: Number of loading workers
        avg_image_load_time_ms: Average time to load one image in milliseconds
    
    Returns:
        Estimated loading time in seconds
    """
    total_images = num_episodes * images_per_episode
    total_time_ms = total_images * avg_image_load_time_ms
    
    if parallel_loading and num_episodes > 2:
        # Parallel loading speedup (not linear due to overhead)
        speedup = min(loading_workers, num_episodes) * 0.7
        total_time_ms /= speedup
    
    return total_time_ms / 1000.0
