# 📘 BICSNet: Bilateral Inertia-Correction Sparse Network

**Deep learning model for correcting particle inertia bias in shock-dominated supersonic PIV data.**

This repository contains the official implementation of **BICSNet (Bilateral Inertia-Correction Sparse Network)**, a deep learning framework trained on synthetic PIV images of oblique shocks and validated on experimental supersonic PIV data from the FSU Polysonic Wind Tunnel.

BICSNet corrects **particle inertia bias** in PIV measurements, improving the fidelity of velocity fields across shocks and enhancing agreement with CFD validation datasets.

---

## 🔍 Overview

- **Problem**: Tracer particles in supersonic flows lag behind fluid motion due to inertia, causing systematic velocity errors across shocks.  
- **Solution**: A physics-aware CNN architecture (bilateral + U-Net inspired) trained on synthetic datasets, conditioned on Mach and Reynolds numbers, to reduce inertia bias.  
- **Contributions**:
  - Open-source model capable of reducing particle inertia bias was provided.
  - A test shock (Mach 2.5) was provided that was not used in the training process to explain the workflow.

---

## 📂 Repository Structure

```
BICSNet-PIV/
│── data/
│   └── test_images/
│       ├── snap1/                   # Input image A (*.tif)
│       ├── particle/                # Input image B (*.tif)
│       ├── fluid/                   # Ground truth (*.tif)
│       └── scalars.csv              # Optional (mach, reynolds number)
│
│── checkpoints/
│   └── best_model.pth               # Trained BICSNet weights
│
│── src/
│   ├── bicsnet.py                   # Model definition
│   ├── loader.py                    # Dataset + transforms
│   └── pivnet_image_gen.py          # Inference and image generation
│
│── analysis.ipynb                   # Optional notebook
│── README.md                        # This file
│── pyproject.toml                   # Project configuration (uv)
│── .python-version                  # Pinned Python version (3.12)
```

---

## ⚙️ Installation (uv + Python 3.12)

This repo uses `uv` for environment management and locking.

```bash
# 1) Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Clone and enter the project
git clone https://github.com/<your-username>/BICSNet-PIV.git
cd BICSNet-PIV

# 3) Create a clean Python 3.12 virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# 4) Install all dependencies (including OpenPIV)
uv sync

# 5) Install the correct PyTorch build for your hardware (GPU/CPU)
# For NVIDIA GPUs with CUDA support:
uv sync --group cuda

# For CPU-only or Apple Silicon (MPS):
uv sync --group cpu

# Or install both and let the system auto-detect (default):
uv sync

# 5) Launch Jupyter
jupyter lab
```

Notes:
- Python pinned to 3.12 for PyTorch compatibility.
- If you see NumPy ABI warnings with PyTorch, use `uv pip install "numpy<2"`.
- Use `uv sync --group cuda` for NVIDIA GPU systems, `uv sync --group cpu` for CPU-only or Apple Silicon.
- Default `uv sync` installs standard PyTorch (MPS-enabled on Apple Silicon, CPU on others).

**Dependencies:**
- Python 3.12
- PyTorch (via dependency groups): `torch`, `torchvision`
- Scientific computing: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-image`, `scikit-learn`
- PIV analysis: `openpiv`
- Utilities: `seaborn`, `tqdm`, `tifffile`, `pillow`

---

## 📊 Dataset

### Synthetic Data (syPIV + LPT)
- 600 training cases (4 Mach numbers × 3 deflection angles × particle specs).  
- 300 testing cases (2 Mach numbers × 3 deflection angles).  
- Each case → 128 image pairs (snapshots + ground truth).  

### Experimental Data (FSU PSWT)
- Mach 2 shock-interaction PIV dataset.  
- Cropped into 256×256 patches with intensity normalization.  
- Full dataset cannot be shared, but representative samples are included.  

---

Example command:  
```bash
python train.py \
  --data ./data/synthetic/ \
  --epochs 100 \
  --batch_size 36 \
  --lr 1e-4 \
  --checkpoint ./checkpoints/bicsnet.pth
```

Training details (as in paper):  
- Optimizer: Adam  
- Loss: Mean Squared Error (MSE)  
- Learning rate: 1e-4 with decay (factor 0.8, patience=2)  
- Batch size: 36  
- Hardware: 4 × NVIDIA H100 GPUs, ~11 days runtime for 77 epochs  

---

## 🔍 Inference (Image Generation)

Generate model outputs for all images in `data/test_images/`:

```bash
source .venv/bin/activate
# uses local checkpoints/best_model.pth if present; otherwise auto-downloads from HF
python src/pivnet_image_gen.py
```

Behavior:
- Automatically selects device: CUDA > MPS (Apple Silicon) > CPU (Intel defaults to CPU).
- Loads checkpoint from `checkpoints/best_model.pth`.
- Saves outputs to `data/test_images/model_outputs1/` and `model_outputs2/`.

### Using the public Hugging Face checkpoint

The model weights are hosted publicly:

- Hugging Face repo: [`kalagotla/BICSNet`](https://huggingface.co/kalagotla/BICSNet/tree/main)

If `checkpoints/best_model.pth` is missing, the script will attempt to download `best_model.pth` from the repo above automatically. You can also specify/override explicitly:

```bash
python src/pivnet_image_gen.py \
  --checkpoint ./checkpoints/best_model.pth \
  --hf-repo kalagotla/BICSNet \
  --hf-file best_model.pth
```

---

## 📑 Results

| Case                  | Metric   | PIV Error | BICSNet Error | Improvement |
|-----------------------|----------|-----------|----------------|-------------|
| Mach 2 (Exp)          | L2-norm | 1.08      | 0.50           | **53.7%**   |
| Mach 2.5 (Synthetic)  | L2-norm | 0.86      | 0.19           | **78%**     |
| Mach 7.6 (OOD)        | L2-norm | 1.81      | 1.27           | **30%**     |

---

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 📌 Citation

If you use this code or datasets, please cite:

```bibtex
@software{kalagotla2025bicsnetpiv,
  title = {BICSNet-PIV: Bilateral Inertia-Correction Sparse Network},
  author = {Kalagotla, Dilip and Cuppoletti, Daniel and Orkwis, Paul and Hernandez-Lichtl, Kevin and Gustavsson, Jonas and Kumar, Rajan},
  year = {2025},
  url = {https://github.com/kalagotla/BICSNet-PIV},
  note = {GitHub repository. MIT License.}
}
```

---

## 🙏 Acknowledgments

- Florida State University (FCAAP) for experimental datasets  
- University of Cincinnati ARC for GPU resources  
- OpenPIV developers for their analysis toolkit
