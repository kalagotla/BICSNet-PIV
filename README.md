# 📘 BICSNet: Bilateral Inertia-Correction Sparse Network

**Deep learning model for correcting particle inertia bias in shock-dominated supersonic PIV data.**

This repository contains the official implementation of **BICSNet (Bilateral Inertia-Correction Sparse Network)**, a deep learning framework trained on synthetic PIV images of oblique shocks and validated on experimental supersonic PIV data from the FSU Polysonic Wind Tunnel.

BICSNet corrects **particle inertia bias** in PIV measurements, improving the fidelity of velocity fields across shocks and enhancing agreement with CFD validation datasets.

---

## 🔍 Overview

- **Problem**: Tracer particles in supersonic flows lag behind fluid motion due to inertia, causing systematic velocity errors across shocks.  
- **Solution**: A physics-aware CNN architecture (bilateral + U-Net inspired) trained on synthetic datasets, conditioned on Mach and Reynolds numbers, to reduce inertia bias.  
- **Contributions**:
  - Open-source code for training and inference.
  - Representative synthetic datasets and sample experimental patches.
  - Preprocessing scripts for experimental PIV data.
  - Trained model checkpoints for reproducibility.

---

## 📂 Repository Structure

```
BICSNet_module/
│── data/                # Sample synthetic & experimental datasets
│   ├── synthetic/       # syPIV-generated images (subset)
│   └── experimental/    # Representative pre-processed PIV patches
│
│── model/               # Network definitions
│── checkpoints/         # Pre-trained BICSNet weights
│── scripts/             # Preprocessing & figure reproduction scripts
│   ├── preprocess_exp.py
│   ├── reproduce_fig22.py
│   └── reproduce_fig24.py
│
│── train.py             # Training loop
│── evaluate.py          # Model evaluation
│── requirements.txt     # Dependencies
│── README.md            # This file
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/BICSNet_module.git
cd BICSNet_module
conda create -n bicsnet python=3.10
conda activate bicsnet
pip install -r requirements.txt
```

**Dependencies:**  
- Python ≥ 3.10  
- PyTorch ≥ 2.0  
- OpenPIV  
- NumPy, SciPy, Matplotlib, scikit-image, pandas  
- tqdm  

---

## 📊 Dataset

### Synthetic Data (syPIV + LPT)
- 600 training cases (4 Mach numbers × 3 deflection angles × particle specs).  
- 300 testing cases (2 Mach numbers × 3 deflection angles).  
- Each case → 128 image pairs (snapshots + ground truth).  

➡️ **Representative dataset samples:** [Zenodo link / Google Drive placeholder]

### Experimental Data (FSU PSWT)
- Mach 2 shock-interaction PIV dataset.  
- Cropped into 256×256 patches with intensity normalization.  
- Full dataset cannot be shared, but representative samples are included.  

---

## 🏋️ Training

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

## 🔍 Evaluation

Evaluate on synthetic test set:  
```bash
python evaluate.py \
  --data ./data/synthetic/test/ \
  --checkpoint ./checkpoints/bicsnet.pth
```

Reproduce **Figure 22 (PIV vs BICSNet vs CFD)**:  
```bash
python scripts/reproduce_fig22.py
```

Reproduce **Figure 24 (Shock interaction)**:  
```bash
python scripts/reproduce_fig24.py
```

---

## 📑 Results

| Case                  | Metric   | PIV Error | BICSNet Error | Improvement |
|-----------------------|----------|-----------|----------------|-------------|
| Mach 2 (Exp)          | L2-norm | 1.08      | 0.50           | **53.7%**   |
| Mach 2.5 (Synthetic)  | L2-norm | 0.86      | 0.19           | **78%**     |
| Mach 7.6 (OOD)        | L2-norm | 1.81      | 1.27           | **30%**     |

---

## 🧩 Reproducibility Checklist

✔ Model architecture & assumptions documented  
✔ Dataset splits & preprocessing described  
✔ Hyperparameters & training runs specified  
✔ Evaluation metrics & scripts provided  
✔ Trained models included  
✔ Hardware/runtime reported  

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 📌 Citation

If you use this code or datasets, please cite:

```bibtex
@article{kalagotla2025bicsnet,
  title={Deep Learning Based Particle Inertia Bias Corrector for Shock-Dominated PIV Data},
  author={Kalagotla, Dilip and Cuppoletti, Daniel and Orkwis, Paul and Hernandez-Lichtl, Kevin and Gustavsson, Jonas and Kumar, Rajan},
  journal={Experiments in Fluids},
  year={2025},
  note={Submitted}
}
```

---

## 🙏 Acknowledgments

- Florida State University (FCAAP) for experimental datasets  
- University of Cincinnati ARC for GPU resources  
- OpenPIV developers for their analysis toolkit
