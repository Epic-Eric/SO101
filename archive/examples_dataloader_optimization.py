"""
Example script showing how to use the optimized data loading API.

This demonstrates the new parallel loading features without requiring
actual data or dependencies.
"""

# Example 1: Basic usage with default parallel loading
def example_basic_usage():
    """
    Basic world model training with default parallel loading enabled.
    """
    print("Example 1: Basic Usage")
    print("-" * 60)
    
    command = """
    python train_world_model.py \\
        data/captured_images_and_joints \\
        output/ \\
        --epochs 100 \\
        --batch_size 16
    """
    print(command)
    print("\nParallel loading is ENABLED by default with 4 workers.")
    print()


# Example 2: Optimized for local SSD storage
def example_local_ssd():
    """
    Optimized configuration for local SSD storage (best performance).
    """
    print("Example 2: Local SSD Storage (Best Performance)")
    print("-" * 60)
    
    command = """
    python train_world_model.py \\
        data/captured_images_and_joints \\
        output/ \\
        --num_workers 4 \\
        --loading_workers 8 \\
        --prefetch_factor 4 \\
        --cache_images \\
        --cache_size 4096
    """
    print(command)
    print("\nBenefits:")
    print("  - High parallelism for fast I/O")
    print("  - Caching for frequently accessed images")
    print("  - Expected speedup: 3-5x during data loading")
    print()


# Example 3: Optimized for network/cloud storage
def example_network_storage():
    """
    Optimized configuration for network or cloud storage (e.g., Google Drive).
    """
    print("Example 3: Network/Cloud Storage")
    print("-" * 60)
    
    command = """
    python train_world_model.py \\
        /mnt/drive/data \\
        output/ \\
        --num_workers 0 \\
        --loading_workers 2 \\
        --preload_images \\
        --preload_dtype float16
    """
    print(command)
    print("\nBenefits:")
    print("  - Minimal network requests")
    print("  - All data loaded once at startup")
    print("  - Reduced memory with float16")
    print("  - Avoids network bottlenecks")
    print()


# Example 4: Large dataset with memory constraints
def example_large_dataset():
    """
    Configuration for large datasets that don't fit in memory.
    """
    print("Example 4: Large Dataset (Memory Constrained)")
    print("-" * 60)
    
    command = """
    python train_world_model.py \\
        data/large_dataset \\
        output/ \\
        --num_workers 2 \\
        --loading_workers 4 \\
        --cache_images \\
        --cache_size 2048 \\
        --prefetch_factor 2
    """
    print(command)
    print("\nBenefits:")
    print("  - LRU cache for hot data")
    print("  - Controlled memory usage")
    print("  - Balanced I/O and compute")
    print()


# Example 5: Disable parallel loading (compatibility mode)
def example_disable_parallel():
    """
    Disable parallel loading for compatibility or debugging.
    """
    print("Example 5: Disable Parallel Loading")
    print("-" * 60)
    
    command = """
    python train_world_model.py \\
        data/captured_images_and_joints \\
        output/ \\
        --no_parallel_loading \\
        --num_workers 0
    """
    print(command)
    print("\nUse when:")
    print("  - Debugging data loading issues")
    print("  - Thread compatibility problems")
    print("  - Network filesystems behave better with sequential access")
    print()


# Example 6: Using configuration file
def example_config_file():
    """
    Set optimization parameters in config.yml.
    """
    print("Example 6: Configuration File")
    print("-" * 60)
    
    config = """
    # config.yml
    world_num_workers: 4
    world_loading_workers: 8
    world_prefetch_factor: 4
    world_cache_size: 4096
    world_preload_dtype: float16
    """
    print(config)
    print("\nThen simply run:")
    print("    python train_world_model.py data_dir output_dir")
    print()


# Example 7: Using Python API directly
def example_python_api():
    """
    Use the ImageJointSequenceDataset API directly in Python.
    """
    print("Example 7: Python API Usage")
    print("-" * 60)
    
    code = """
    from model.src.interfaces.dataset import ImageJointSequenceDataset
    
    # Create dataset with parallel loading
    dataset = ImageJointSequenceDataset(
        root_dir="data/captured_images_and_joints",
        seq_len=16,
        image_size=64,
        parallel_loading=True,      # Enable parallel loading
        loading_workers=8,           # Use 8 worker threads
        preload_images=False,        # Don't preload (for large datasets)
        cache_images=True,           # Enable LRU cache
        cache_size=4096,             # Cache up to 4096 images
    )
    
    print(f"Loaded {dataset.num_episodes} episodes")
    print(f"Total sequences: {len(dataset)}")
    """
    print(code)
    print()


# Example 8: Using optimization utilities
def example_utilities():
    """
    Use helper utilities to get optimal configuration.
    """
    print("Example 8: Optimization Utilities")
    print("-" * 60)
    
    code = """
    from model.src.utils.dataloader_utils import (
        get_optimal_workers,
        get_cache_recommendations,
        print_optimization_tips
    )
    
    # Get optimal worker configuration
    num_workers, loading_workers = get_optimal_workers(
        storage_type="local",  # "local", "network", or "cloud"
        num_episodes=50
    )
    print(f"Recommended: num_workers={num_workers}, loading_workers={loading_workers}")
    
    # Get cache recommendations
    cache_config = get_cache_recommendations(
        num_images=10000,
        image_size=64,
        available_memory_gb=16.0
    )
    print(f"Cache config: {cache_config}")
    
    # Print comprehensive optimization tips
    print_optimization_tips(
        num_episodes=50,
        num_images=10000,
        storage_type="local"
    )
    """
    print(code)
    print()


def main():
    """Show all examples."""
    print("\n" + "="*70)
    print("DATA LOADING OPTIMIZATION - USAGE EXAMPLES")
    print("="*70)
    print()
    
    examples = [
        example_basic_usage,
        example_local_ssd,
        example_network_storage,
        example_large_dataset,
        example_disable_parallel,
        example_config_file,
        example_python_api,
        example_utilities,
    ]
    
    for example in examples:
        example()
    
    print("="*70)
    print("For more details, see docs/DATA_LOADING_OPTIMIZATION.md")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
