# Notebooks Overview

Progressive notebooks for thesis implementation.

## Getting Started
Start with `00_getting_started.ipynb` to verify setup.

## Week 3: Data Exploration
**Notebook:** `01_data_exploration.ipynb`
- Load COSOCO dataset
- Visualize samples (benign vs malicious)
- Check ground-truth masks
- Understand Hilbert curve encoding
- Implement patch extraction
- Verify data pipeline

## Week 4: Baseline Training
**Notebook:** `02_baseline_training.ipynb`
- Implement ResNet18
- Setup training loop
- Train for 30 epochs
- Target: F1 ≈ 0.736
- Save best checkpoint
- Plot training curves

## Week 5-6: XAI Analysis
**Notebook:** `03_xai_analysis.ipynb`
- Load trained ResNet18
- Implement GradCAM
- Generate heatmaps for samples
- Visualize overlays
- Compute IoU with ground-truth
- Compare benign vs malicious patterns

## Week 9-10: Hilbert Inverse
**Notebook:** `04_hilbert_inverse.ipynb`
- Implement inverse mapping
- Map heatmap → byte ranges
- Map bytes → files
- Generate forensic reports
- Case studies

## Tips

- Run notebooks on Kaggle (free T4 GPU)
- Save outputs to `outputs/` directory
- Document observations in markdown cells
- Keep clean outputs for thesis figures
