"""
Test script to verify data loading optimizations work correctly.

This script creates mock data and tests the parallel loading functionality.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_episode(episode_dir: Path, num_frames: int = 10):
    """Create a mock episode with joints.jsonl and dummy image references."""
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Create joints.jsonl
    joints_data = []
    for i in range(num_frames):
        frame_data = {
            "t": i * 0.1,
            "image": f"frame_{i:06d}.jpg",
            "joints": {
                "joint_0": float(i) * 0.1,
                "joint_1": float(i) * 0.2,
                "joint_2": float(i) * 0.3,
            }
        }
        joints_data.append(frame_data)
    
    with open(episode_dir / "joints.jsonl", "w") as f:
        for record in joints_data:
            f.write(json.dumps(record) + "\n")
    
    # Create dummy images (we'll use PIL to create actual images)
    try:
        from PIL import Image
        import numpy as np
        
        for i in range(num_frames):
            # Create a simple gradient image
            img_array = np.ones((64, 64, 3), dtype=np.uint8) * (i * 25 % 256)
            img = Image.fromarray(img_array, 'RGB')
            img.save(episode_dir / f"frame_{i:06d}.jpg")
    except ImportError:
        print("PIL not available, skipping image creation")
        return False
    
    return True


def test_episode_discovery():
    """Test parallel episode discovery."""
    print("\n" + "="*60)
    print("TEST: Episode Discovery")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create mock episodes
        num_episodes = 5
        for i in range(num_episodes):
            ep_dir = root / f"episode_{i:03d}"
            success = create_mock_episode(ep_dir, num_frames=15)
            if not success:
                print("⚠️  Skipping test - PIL not available")
                return False
        
        print(f"✓ Created {num_episodes} mock episodes")
        
        # Test discovery
        try:
            from model.src.interfaces.dataset import _discover_episode_dirs
            
            # Test sequential
            episodes_seq = _discover_episode_dirs(str(root), parallel=False)
            print(f"✓ Sequential discovery found {len(episodes_seq)} episodes")
            
            # Test parallel
            episodes_par = _discover_episode_dirs(str(root), parallel=True, max_workers=4)
            print(f"✓ Parallel discovery found {len(episodes_par)} episodes")
            
            assert len(episodes_seq) == num_episodes, f"Expected {num_episodes}, got {len(episodes_seq)}"
            assert len(episodes_par) == num_episodes, f"Expected {num_episodes}, got {len(episodes_par)}"
            assert sorted(episodes_seq) == sorted(episodes_par), "Sequential and parallel results differ"
            
            print("✓ Episode discovery tests passed!")
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_dataset_loading():
    """Test dataset loading with parallel option."""
    print("\n" + "="*60)
    print("TEST: Dataset Loading")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create mock episodes
        num_episodes = 3
        for i in range(num_episodes):
            ep_dir = root / f"episode_{i:03d}"
            success = create_mock_episode(ep_dir, num_frames=20)
            if not success:
                print("⚠️  Skipping test - PIL not available")
                return False
        
        print(f"✓ Created {num_episodes} mock episodes with 20 frames each")
        
        try:
            from model.src.interfaces.dataset import ImageJointSequenceDataset
            import time
            
            # Test sequential loading
            print("\nTesting sequential loading...")
            t0 = time.time()
            dataset_seq = ImageJointSequenceDataset(
                root_dir=str(root),
                seq_len=5,
                image_size=64,
                parallel_loading=False,
                loading_workers=1,
                preload_images=False,
            )
            t_seq = time.time() - t0
            print(f"✓ Sequential loading took {t_seq:.3f}s")
            print(f"  - Episodes: {dataset_seq.num_episodes}")
            print(f"  - Sequences: {len(dataset_seq)}")
            
            # Test parallel loading
            print("\nTesting parallel loading...")
            t0 = time.time()
            dataset_par = ImageJointSequenceDataset(
                root_dir=str(root),
                seq_len=5,
                image_size=64,
                parallel_loading=True,
                loading_workers=4,
                preload_images=False,
            )
            t_par = time.time() - t0
            print(f"✓ Parallel loading took {t_par:.3f}s")
            print(f"  - Episodes: {dataset_par.num_episodes}")
            print(f"  - Sequences: {len(dataset_par)}")
            
            speedup = t_seq / t_par if t_par > 0 else 1.0
            print(f"\n✓ Speedup: {speedup:.2f}x")
            
            # Verify both produce same results
            assert dataset_seq.num_episodes == dataset_par.num_episodes
            assert len(dataset_seq) == len(dataset_par)
            
            print("✓ Dataset loading tests passed!")
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_utilities():
    """Test utility functions."""
    print("\n" + "="*60)
    print("TEST: Utility Functions")
    print("="*60)
    
    try:
        from model.src.utils.dataloader_utils import (
            get_optimal_workers,
            get_cache_recommendations,
            estimate_loading_time
        )
        
        # Test get_optimal_workers
        nw, lw = get_optimal_workers("local", num_episodes=50)
        print(f"✓ get_optimal_workers('local', 50): num_workers={nw}, loading_workers={lw}")
        
        nw, lw = get_optimal_workers("network", num_episodes=10)
        print(f"✓ get_optimal_workers('network', 10): num_workers={nw}, loading_workers={lw}")
        
        # Test cache recommendations
        cache_config = get_cache_recommendations(num_images=1000, image_size=64)
        print(f"✓ get_cache_recommendations(1000 images): {cache_config['reason']}")
        
        # Test time estimation
        estimated_time = estimate_loading_time(
            num_episodes=50,
            images_per_episode=200,
            parallel_loading=True,
            loading_workers=4
        )
        print(f"✓ estimate_loading_time: ~{estimated_time:.1f}s")
        
        print("✓ Utility function tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DATA LOADING OPTIMIZATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Episode Discovery", test_episode_discovery()))
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Utility Functions", test_utilities()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
