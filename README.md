# ShiftGuard10: Robust Image Classification

A deep learning framework for **distribution-shift robust image classification** using a **WideResNet-40-10 ensemble** with advanced augmentation, imbalance handling, and test-time robustness techniques.

## Overview

ShiftGuard10 is designed to tackle two major real-world computer vision challenges:

- **Severe class imbalance** (up to 50:1 ratio)
- **Distribution shift** between training and testing images

The project uses a **3-seed WideResNet-40-10 ensemble** combined with:
- BalancedSoftmaxLoss
- Stochastic Weight Averaging (SWA)
- Test-Time Augmentation (TTA)
- MixUp & CutMix
- Custom GreyCutout augmentation

The system achieved a **best validation Macro F1-score of 0.9206** and targets **F1 > 0.95** with ensemble inference.

---

## Features

- WideResNet-40-10 backbone (~54M parameters)
- Distribution-shift robust training pipeline
- Advanced augmentation strategies
- Ensemble learning with multiple random seeds
- Mixed precision GPU training (AMP)
- SWA-based model averaging
- 30-round Test-Time Augmentation
- Robust handling of long-tail class imbalance

---

## Dataset

- CIFAR-style 10-class image dataset
- Total Images: **29,400**
- Image Size: **32×32 RGB**
- Severe imbalance ratio: **50:1**

### Classes
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## Architecture

### Backbone
- **WideResNet-40-10**
- Depth: 40
- Width Factor: 10
- Parameters: ~54M

### Ensemble Strategy
- 3 independent random seeds
- Probability averaging during inference

---

## Techniques Used

### Imbalance Handling
- Weighted Random Sampler
- BalancedSoftmaxLoss
- Label Smoothing

### Data Augmentation
- AutoAugment
- AugMix
- MixUp
- CutMix
- Random Erasing
- Gaussian Noise
- Custom GreyCutout

### Optimization
- SGD + Nesterov Momentum
- Cosine Annealing LR Scheduler
- Stochastic Weight Averaging (SWA)
- Automatic Mixed Precision (AMP)

---

## Results

| Metric | Value |
|---|---|
| Best Validation Macro F1 | **0.9206** |
| Target Ensemble F1 | **> 0.95** |
| Training Epochs | 300 |
| Ensemble Seeds | 3 |
| TTA Rounds | 30 per model |

---

## Project Structure

```bash
├── notebooks/
├── models/
├── training/
├── augmentation/
├── checkpoints/
├── inference/
├── report/
├── presentation/
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/ShiftGuard10.git
cd ShiftGuard10
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Inference

```bash
python inference.py
```

---

## Key Learnings

- Layered robustness strategies outperform single-technique solutions
- SWA and TTA significantly improve calibration under distribution shift
- BalancedSoftmaxLoss effectively handles long-tail distributions
- Custom augmentations can target dataset-specific corruptions

---

## Authors

- Aarnav Gupta
- Aarush Singh
- Kaushey
- Prince Pathak
- Sushant Kumar

Indian Institute of Technology Kanpur

---

## Course Information

**Course:** EE708  
**Instructor:** Prof. Rajesh M. Hegde  
Department of Electrical Engineering  
IIT Kanpur

---

## License

This project is for academic and educational purposes.
