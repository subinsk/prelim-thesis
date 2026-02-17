# Running Mid-Semester Experiments

Complete guide for running all experiments and collecting results for mid-semester evaluation.

## ✅ Pre-Flight Checklist

All code is **100% implemented** and ready to run:

- ✅ Data pipeline (dataset, transforms, dataloaders)
- ✅ Models (ResNet18, Swin Transformer)
- ✅ Training script with early stopping and checkpointing
- ✅ Evaluation script with metrics and visualizations
- ✅ GradCAM XAI implementation
- ✅ Comprehensive metrics (classification + XAI)
- ✅ Visualization utilities

## Step 1: Activate Virtual Environment

```bash
cd d:/projects/prelim-thesis

# If you haven't created venv yet:
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 3: Download COSOCO Dataset

```bash
# This will download ~4GB of data from HuggingFace
python scripts/download_dataset.py
```

**Expected output:**
```
data/cosoco/
├── train/
│   ├── benign/ (1,557 images)
│   └── compromised/ (797 images)
├── val/
│   ├── benign/ (334 images)
│   └── compromised/ (171 images)
└── test/
    ├── benign/ (334 images)
    └── compromised/ (171 images)
```

## Step 4: Train ResNet18 Baseline

**Target: Match or beat F1 = 0.736 from paper**

```bash
# Basic training (30 epochs)
python experiments/train.py --config configs/config.yaml --model resnet18 --epochs 30

# With WandB logging (optional)
python experiments/train.py --config configs/config.yaml --model resnet18 --epochs 30 --wandb

# Resume from checkpoint if interrupted
python experiments/train.py --resume outputs/models/resnet18_latest.pth
```

**Training time estimate:**
- CPU: ~8-12 hours for 30 epochs
- GPU (T4/V100): ~45-90 minutes for 30 epochs

**What to expect:**
```
Epoch 1/30
--------------------------------------------------
Training: 100%|████████| 588/588 [12:34<00:00,  1.28s/it]
Validating: 100%|████████| 127/127 [02:15<00:00,  1.06s/it]

Train - Loss: 0.4523, Acc: 0.7821, Prec: 0.7456, Rec: 0.8012, F1: 0.7723
Val   - Loss: 0.3891, Acc: 0.8234, Prec: 0.7998, Rec: 0.8456, F1: 0.8221

*** New best F1: 0.8221 ***
Saved BEST checkpoint to outputs/models/resnet18_best.pth
```

## Step 5: Evaluate Trained Model

```bash
# Evaluate on test set
python experiments/evaluate.py --checkpoint outputs/models/resnet18_best.pth --split test

# Evaluate on validation set
python experiments/evaluate.py --checkpoint outputs/models/resnet18_best.pth --split val
```

**Output:**
- Metrics printed to console
- Confusion matrix saved to `outputs/results/resnet18_confusion_matrix.png`
- ROC curve saved to `outputs/results/resnet18_roc_curve.png`
- Metrics JSON saved to `outputs/results/resnet18_metrics.json`

## Step 6: Train Swin Transformer (Optional for Mid-Sem)

```bash
# Train Swin-Tiny
python experiments/train.py --config configs/swin_config.yaml --model swin_tiny --epochs 30

# Evaluate Swin
python experiments/evaluate.py --checkpoint outputs/models/swin_tiny_best.pth --split test
```

## Step 7: Generate XAI Visualizations

```python
# Quick script to generate GradCAM for a few samples
python -c "
import torch
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from src.models.resnet import ResNet18Classifier
from src.xai.gradcam import GradCAMExplainer, get_target_layer
from src.data.dataset import COSOCODataset
from src.data.transforms import get_val_transforms
from src.utils.visualization import plot_gradcam_comparison
import matplotlib.pyplot as plt

# Load model
checkpoint = torch.load('outputs/models/resnet18_best.pth', map_location='cpu')
model = ResNet18Classifier(num_classes=2, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test dataset
dataset = COSOCODataset('data/cosoco', split='test', transform=get_val_transforms(256))

# Get a compromised sample
for idx in range(len(dataset)):
    sample = dataset[idx]
    if sample['image_label'] == 1:  # Compromised
        break

# Initialize GradCAM
target_layers = [model.backbone.layer4[-1]]
explainer = GradCAMExplainer(model, target_layers, device='cpu')

# Generate heatmaps for first patch
patch = sample['patches'][0]  # First patch
gradcam_maps = {
    'GradCAM': explainer.explain(patch, method='gradcam'),
    'HiResCAM': explainer.explain(patch, method='hirescam'),
}

print('GradCAM heatmaps generated!')
print('Shape:', gradcam_maps['GradCAM'].shape)
"
```

## Expected Results for Mid-Semester

### Minimum Deliverables (Week 8)

**1. ResNet18 Baseline:**
- ✅ Trained model checkpoint
- ✅ F1 score ≈ 0.736 (matching paper)
- ✅ Confusion matrix
- ✅ Training curves

**2. XAI Visualizations:**
- ✅ GradCAM heatmaps for sample images
- ✅ Comparison of compromised vs benign attention patterns

**3. Documentation:**
- ✅ Training logs
- ✅ Evaluation metrics (JSON)

### Bonus (If Time Permits)

- Swin Transformer results
- XAI evaluation metrics (IoU, Pointing Game)
- Comparison: ResNet18 vs Swin

## Quick Commands Cheat Sheet

```bash
# Download dataset
python scripts/download_dataset.py

# Train ResNet18
python experiments/train.py --model resnet18 --epochs 30

# Evaluate
python experiments/evaluate.py --checkpoint outputs/models/resnet18_best.pth

# Train Swin
python experiments/train.py --model swin_tiny --epochs 30

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Issue: Out of memory
**Solution:** Reduce batch size in `configs/config.yaml`:
```yaml
training:
  batch_size: 2  # Reduce from 4 to 2
```

### Issue: CUDA out of memory
**Solution:** Use CPU or reduce max patches:
```yaml
data:
  max_patches_per_image: 50  # Reduce from 100
```

### Issue: Dataset download fails
**Solution:**
```bash
# Manual download from HuggingFace
# Visit: https://huggingface.co/datasets/k3ylabs/cosoco-image-dataset
# Download and extract to data/cosoco/
```

## File Organization After Running

```
prelim-thesis/
├── data/
│   └── cosoco/              # Downloaded dataset
├── outputs/
│   ├── models/
│   │   ├── resnet18_best.pth    # Best model
│   │   └── resnet18_latest.pth  # Latest checkpoint
│   ├── results/
│   │   ├── resnet18_metrics.json
│   │   ├── resnet18_confusion_matrix.png
│   │   └── resnet18_roc_curve.png
│   └── figures/             # XAI visualizations
└── logs/                    # Training logs
```

## Next Steps After Mid-Sem

1. Hilbert inverse mapping implementation
2. SHAP and LIME integration
3. Comprehensive XAI evaluation
4. Forensic case studies
5. Thesis writing

---

**Status:** All code implemented ✅
**Ready to run:** Yes ✅
**Just need:** Dataset download + GPU time
