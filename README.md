# Explainable AI for Docker Container Malware Detection

My thesis work on using Vision Transformers and XAI techniques to detect malware in Docker containers.

**Supervisor:** Dr. Ferdous Ahmed Barbhuiya
**IIIT Guwahati** | Feb - Jun 2026

## What's This About?

I'm building on the [Nousias et al. (2025)](https://arxiv.org/abs/2504.03238) paper that converts Docker containers to images and uses CNNs to detect malware. The problem is - their model is a black box. You have no idea WHY it thinks a container is malicious.

My work adds:
- Explainability (GradCAM, SHAP, LIME) to show what the model is actually looking at
- Vision Transformers (Swin, ViT) - nobody's tried these on container images yet
- A novel way to map the attention back to actual files in the container (using Hilbert curve inverse mapping)

Basically, instead of just "this is malware", we can now say "this is malware because of these specific bytes in /usr/bin/httpd".

## Questions I'm Trying to Answer

1. Do Vision Transformers work better than CNNs for this task?
2. Which XAI method (GradCAM, SHAP, LIME) actually highlights the real malware regions?
3. Can we map model attention back to specific malicious files?
4. Do transformers focus on more useful forensic features than CNNs?

## What I'm Contributing

- First proper XAI evaluation on container malware images (with actual metrics)
- Testing Vision Transformers on containers (nobody's done this yet)
- Inverse Hilbert mapping to go from heatmap → actual files (completely new)
- Something security analysts can actually use in practice

## Dataset

Using **COSOCO** from [HuggingFace](https://huggingface.co/datasets/k3ylabs/cosoco-image-dataset)
- 3,364 images (2,225 clean, 1,139 compromised)
- 10 malware families: Mirai, Gafgyt, CoinMiner, etc.
- Ground-truth masks included (huge win!)
- License: CC-BY-4.0

## 🏗️ Project Structure

```
prelim-thesis/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── configs/                     # Configuration files
│   ├── config.yaml              # Main config
│   ├── resnet_config.yaml       # ResNet-specific
│   └── swin_config.yaml         # Swin-specific
│
├── src/                         # Source code
│   ├── data/                    # Dataset and dataloaders
│   ├── models/                  # Model architectures
│   ├── xai/                     # XAI implementations
│   ├── hilbert/                 # Hilbert mapping
│   └── utils/                   # Utilities
│
├── experiments/                 # Training scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── explain.py               # XAI generation
│
├── notebooks/                   # Jupyter notebooks
├── thesis/                      # LaTeX thesis
├── outputs/                     # Results and models
└── docs/                        # Documentation
```

## Setup

Need:
- Python 3.9+
- GPU with 16GB VRAM (using Kaggle T4 for now)
- 16GB RAM minimum

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/prelim-thesis.git
cd prelim-thesis

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install everything
pip install -r requirements.txt
```

Get the dataset:
```bash
from datasets import load_dataset
dataset = load_dataset('k3ylabs/cosoco-image-dataset')
```

## 💻 Usage

### Training a Model

```bash
# Train ResNet18 baseline
python experiments/train.py --config configs/resnet_config.yaml

# Train Swin Transformer
python experiments/train.py --config configs/swin_config.yaml

# Train with custom parameters
python experiments/train.py --model resnet18 --epochs 50 --batch-size 8 --lr 0.0001
```

### Evaluation

```bash
# Evaluate on test set
python experiments/evaluate.py --checkpoint outputs/models/best_resnet18.pth

# Generate confusion matrix and metrics
python experiments/evaluate.py --checkpoint outputs/models/swin_tiny.pth --test-only
```

### Generate XAI Explanations

```bash
# Generate GradCAM visualization
python experiments/explain.py --checkpoint outputs/models/best_resnet18.pth --method gradcam --sample-idx 0

# Generate all XAI methods
python experiments/explain.py --checkpoint outputs/models/swin_tiny.pth --method all

# Generate forensic report
python experiments/explain.py --checkpoint outputs/models/best_resnet18.pth --forensic
```

## 📈 Results (To Be Updated)

### Classification Performance

| Model           | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| ResNet18       | TBD%     | TBD       | TBD    | TBD      |
| Swin-Tiny      | TBD%     | TBD       | TBD    | TBD      |
| ViT-Small      | TBD%     | TBD       | TBD    | TBD      |

### XAI Evaluation

| Method    | Mean IoU | Pointing Game Acc |
|-----------|----------|-------------------|
| GradCAM   | TBD      | TBD%             |
| HiResCAM  | TBD      | TBD%             |
| SHAP      | TBD      | TBD%             |
| LIME      | TBD      | TBD%             |

## 📝 Thesis Writing

The LaTeX thesis is in the `thesis/` directory:

```bash
# Compile thesis
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Timeline

**Phase 1 (Weeks 1-8):** Mid-sem prep
- Literature review
- Get ResNet18 baseline working (match paper's F1 ≈ 0.736)
- Train Swin-Tiny
- Basic GradCAM visualizations

**Phase 2 (Weeks 9-16):** End-sem
- Hilbert inverse mapping (the hard part)
- SHAP and LIME
- Forensic case studies
- All experiments and write-up

## 📚 Key References

1. Nousias et al. (2025) - "Container Malware Detection via Image-Based Deep Learning" [arXiv:2504.03238]
2. Nataraj et al. (2011) - "Malware Images: Visualization and Automatic Classification"
3. Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"
4. Liu et al. (2021) - "Swin Transformer: Hierarchical Vision Transformer"

## Thanks

- Dr. Ferdous Ahmed Barbhuiya (supervisor)
- K3yLabs for releasing COSOCO dataset
- Nousias et al. for the base paper

## Links

- [COSOCO Dataset](https://huggingface.co/datasets/k3ylabs/cosoco-image-dataset)
- [Base Paper](https://arxiv.org/abs/2504.03238)
- [PyTorch GradCAM](https://github.com/jacobgil/pytorch-grad-cam)

---

**Status:** Setting up (Feb 2026)
