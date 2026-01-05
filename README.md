# SO101 - Sim-to-Real Robot Learning

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Epic-Eric/SO101.svg?style=social&label=Star)](https://github.com/Epic-Eric/SO101)
[![GitHub forks](https://img.shields.io/github/forks/Epic-Eric/SO101.svg?style=social&label=Fork)](https://github.com/Epic-Eric/SO101)
[![Last Commit](https://img.shields.io/github/last-commit/Epic-Eric/SO101)](https://github.com/Epic-Eric/SO101/commits)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

A robotics project focused on sim-to-real transfer learning using Variational Autoencoders (VAE) for visual representation learning and teleoperation data collection.

SO101 aims to provide a practical, reproducible pipeline for sim-to-real robot learning experiments, from data collection to world-model visualization.

> **Status:** Work in progress; APIs, configs, and workflows may change as development continues.

## 🚀 Features

- **Data Collection**: Teleoperation-based image capture system for robot learning
- **VAE Training**: Train variational autoencoders on collected images for representation learning
- **LeRobot Integration**: Includes [HuggingFace LeRobot](https://github.com/huggingface/lerobot) as a submodule
- **Flexible Configuration**: YAML-based configuration for easy experimentation
- **Multi-Device Support**: Automatic device selection (CPU, CUDA, MPS)
- **World Model Workflow**: Collect synchronized images + joints, train an RSSM world model, and visualize rollouts

## 📋 Repository Structure

```
SO101/
├── collect_data.py         # Script for collecting teleoperation data
├── collect_image_and_joint.py # Capture synchronized camera frames + joints.jsonl
├── train_model.py          # Main training script for VAE
├── train_world_model.py    # Train VAE + RSSM world model on image/joint sequences
├── world_model_inference.py # Interactive world-model rollout viewer (pygame/headless)
├── run_app.py              # Launcher for Streamlit apps (reward camera, world-model collector)
├── config.yml              # Configuration file for training parameters
├── requirements.txt        # Python dependencies
├── model/                  # Robot model package
│   ├── src/
│   │   ├── core/          # Core training logic and orchestration helpers
│   │   ├── data/          # Data collection modules
│   │   ├── interfaces/    # Type definitions and interfaces
│   │   ├── models/        # VAE model implementations
│   │   └── utils/         # Utility functions (config, IO, etc.)
│   └── visualization/     # Visualization tools
├── data/                   # Directory for collected data
├── output/                 # Training outputs and checkpoints
├── test_use/              # Test utilities and examples
│   ├── camera/            # Camera testing
│   └── teleop/            # Teleoperation testing
└── lerobot/               # LeRobot submodule (HuggingFace)
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.7+
- CUDA-compatible GPU (optional, but recommended)

### Setup

1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive https://github.com/Epic-Eric/SO101.git
   cd SO101
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the model package:**
   ```bash
   pip install -e ./model
   ```

## 📖 Usage

### Data Collection

Collect images using teleoperation:

```bash
python collect_data.py
```

This will:
- Start a teleoperation session
- Capture images for 30 seconds (configurable)
- Save images to `data/captured_images/`
- Display a window showing the captured images

### Training VAE

Train a VAE model on collected images:

```bash
python train_model.py [data_dir] [output_dir] [options]
```

**Example:**
```bash
python train_model.py ./data/captured_images ./output --epochs 480 --batch_size 128 --lr 0.002
```

**Arguments:**
- `data_dir`: Directory containing training images (jpg/png)
- `output_dir`: Output directory for checkpoints and reconstructions
- `--epochs`: Number of training epochs (default: from config.yml)
- `--batch_size`: Batch size (default: from config.yml)
- `--lr`: Learning rate (default: from config.yml)
- `--latent_dim`: Latent dimension size (default: from config.yml)

### Streamlit Apps

Use the unified launcher to run bundled Streamlit tools:

```bash
python run_app.py reward_camera   # Red bead detector + teleop UI
python run_app.py world_collect   # World-model data collection UI
```

`run_app.py` automatically adds the repository root to `PYTHONPATH`, ensuring absolute imports work regardless of your current working directory when launching the command.

### World Model Workflow

**1) Collect synchronized images + joints**

- Set hardware env vars (e.g., `export FOLLOWER_PORT=/dev/ttyUSB0`; optional `export LEADER_PORT=/dev/ttyUSB1` for teleoperation follow mode).
- Run:
  
  ```bash
  python collect_image_and_joint.py
  ```
- Episodes are saved under `data/captured_images_and_joints/episode_*/` with `joints.jsonl` and JPEG frames.

**2) Train the world model (VAE + RSSM)**

```bash
python train_world_model.py data/captured_images_and_joints ./output --seq_len 16 --image_size 64
```

Arguments fall back to `config.yml`:
- `data_dir` uses `world_data_dir` from config.yml, falling back to the general `data_dir` if `world_data_dir` is not configured.
- `out_dir` uses `world_out_dir`; if `world_out_dir` is not configured, it uses the general `out_dir` value from config.yml.
- Hyperparameters honor `world_*` overrides such as `world_epochs`, `world_batch_size`, `world_lr`, and `world_latent_dim`.

**3) Visualize rollouts / reconstructions**

```bash
python world_model_inference.py --data-root data/captured_images_and_joints --artifact-dir output/artifacts
# Add --headless to dump PNGs without opening a pygame window
```

### Configuration

Default parameters can be set in `config.yml`:

```yaml
data_dir: "./data"
session_name: "test_session"
out_dir: "output/"
epochs: 480
batch_size: 128
lr: 0.002
latent_dim: 128
```

## 🧠 Model Architecture

The project uses a **Variational Autoencoder (VAE)** architecture for learning compressed representations of visual data:

- **Encoder**: Compresses input images to a latent space
- **Latent Space**: Low-dimensional representation (default: 128 dimensions)
- **Decoder**: Reconstructs images from latent representations

This is useful for:
- Dimensionality reduction
- Feature extraction for robot control
- Sim-to-real transfer learning

## 🎮 Components

### 1. Data Collection (`collect_data.py`)
Uses teleoperation to collect images with metadata, leveraging the `model.src.data.collect_images_with_teleoperation` function.

### 2. VAE Training (`train_model.py`)
Trains a VAE on collected images with:
- Automatic device selection (MPS/CUDA/CPU)
- Checkpoint saving
- Training metrics tracking
- Image reconstruction visualization

### 3. Model Package (`model/`)
A standalone Python package providing:
- VAE model implementations
- Training utilities
- Data loading and preprocessing
- Teleoperation interfaces
- Visualization tools

## 🤖 LeRobot Integration

This project includes [HuggingFace LeRobot](https://github.com/huggingface/lerobot) as a submodule, which provides state-of-the-art tools for robot learning:

- Pre-trained models
- Dataset utilities
- Policy learning algorithms

## 📊 Output

After training, the `output/` directory will contain:

- **Checkpoints**: Model weights saved during training
- **Reconstructions**: Visual comparison of input vs. reconstructed images
- **Metrics**: JSON files with training metrics (loss, etc.)
- **Final Model**: Best performing model weights

## 🔧 Development

### Running Tests

```bash
# Test camera functionality
python test_use/camera/main.py

# Test teleoperation
python test_use/teleop/teleop_api.py
```

## 📝 License

Proprietary - See LICENSE for details

## 👥 Authors

- ericx

## 🙏 Acknowledgments

- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) for robotics learning tools
- PyTorch team for the deep learning framework

## 📞 Support

For questions or issues, please open an issue on the [GitHub repository](https://github.com/Epic-Eric/SO101/issues).

---

**Note**: This project requires appropriate hardware setup (cameras, robot hardware) for full functionality. The training component can run independently with collected image datasets.
