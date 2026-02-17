# Quick Start Guide

Everything you need to get started and reach mid-semester.

## Initial Setup (Do Once)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Check setup
bash scripts/check_setup.sh

# 4. Download dataset (Week 3)
bash scripts/download_data.sh
```

## Daily Workflow

```bash
# Start your session
source venv/bin/activate

# Check what's next
cat docs/MID_SEM_PROGRESS.md

# Run training (when ready)
python experiments/train.py --config configs/resnet_config.yaml

# Open notebooks (on Kaggle or locally)
jupyter notebook
```

## Week-by-Week Roadmap

### Week 1-2: Literature Review ✓
- Read 20+ papers
- Take notes in `docs/literature_notes/`
- See: `docs/WEEK_1_TASKS.md`

### Week 3: Data Pipeline
- Follow: `docs/WEEK_3_GUIDE.md`
- Download COSOCO
- Implement dataset.py
- Complete: `notebooks/01_data_exploration.ipynb`

### Week 4: Baseline Training
- Follow: `docs/WEEK_4_GUIDE.md`
- Train ResNet18
- Target: F1 = 0.736
- Complete: `notebooks/02_baseline_training.ipynb`

### Week 5-6: XAI Implementation
- Implement GradCAM
- Generate visualizations
- Compute IoU scores
- Complete: `notebooks/03_xai_analysis.ipynb`

### Week 7: Vision Transformer
- Train Swin-Tiny
- Compare with ResNet18
- Analyze attention patterns

### Week 8: Mid-Sem Deliverables
- Presentation (15-20 slides)
- Report (15-20 pages)
- Demo of working system

## Key Files to Edit

**Week 3:**
- `src/data/dataset.py` - Dataset class
- `src/data/transforms.py` - Data augmentation

**Week 4:**
- `src/models/resnet.py` - Model architecture
- `src/models/mil_aggregator.py` - MIL pooling
- `experiments/train.py` - Training loop

**Week 5-6:**
- `src/xai/gradcam.py` - GradCAM implementation
- `src/utils/metrics.py` - IoU computation

**Week 7:**
- `src/models/swin.py` - Swin Transformer

## Commands Cheat Sheet

```bash
# Check setup
bash scripts/check_setup.sh

# Train ResNet18
python experiments/train.py --config configs/resnet_config.yaml

# Train Swin-Tiny
python experiments/train.py --config configs/swin_config.yaml

# Evaluate model
python experiments/evaluate.py --checkpoint outputs/models/best_resnet18.pth

# Generate XAI
python experiments/explain.py --checkpoint outputs/models/best_resnet18.pth --method gradcam

# Compile thesis
cd thesis && pdflatex main.tex
```

## Progress Tracking

Check progress:
```bash
cat docs/MID_SEM_PROGRESS.md
```

Update when done with tasks:
```bash
# Edit docs/MID_SEM_PROGRESS.md
# Mark [x] for completed items
```

## Getting Help

1. **Code issues:** Check docstrings in src/ files
2. **Setup issues:** Run `bash scripts/check_setup.sh`
3. **Week guidance:** Read `docs/WEEK_X_GUIDE.md`
4. **Supervisor:** Weekly meetings (schedule in advance)

## Milestones

- ✓ **Week 1:** Setup done
- **Week 3:** Data pipeline working
- **Week 4:** F1 = 0.736 achieved
- **Week 6:** GradCAM visualizations ready
- **Week 7:** Swin-Tiny beats ResNet18
- **Week 8:** Mid-sem evaluation passed

## What's in Each Directory

```
prelim-thesis/
├── configs/        # Training configurations
├── src/           # Core implementation code
├── experiments/   # Training/eval scripts  
├── notebooks/     # Jupyter notebooks for experiments
├── docs/          # Planning docs & guides
├── thesis/        # LaTeX thesis writing
├── outputs/       # Results, models, figures
└── scripts/       # Helper bash scripts
```

## Tips for Success

1. **Don't skip Week 1-2** - Literature review is foundation
2. **Week 3 is crucial** - Data pipeline must be solid
3. **Week 4 target: F1 = 0.736** - Match the paper exactly
4. **Document everything** - Screenshots, observations, metrics
5. **Commit regularly** - Use git after each working feature
6. **Ask supervisor early** - Don't wait until stuck

## Current Status

**Week:** 1 (Setup complete)  
**Next Task:** Start reading papers  
**Blocker:** None

---
For detailed week-by-week instructions, see `docs/WEEK_X_GUIDE.md` files.
