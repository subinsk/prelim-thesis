#!/bin/bash
# Script to run all experiments

echo "Running all experiments for MTech thesis..."

# Train ResNet18
echo "Training ResNet18..."
python experiments/train.py --config configs/resnet_config.yaml --wandb

# Train Swin-Tiny
echo "Training Swin-Tiny..."
python experiments/train.py --config configs/swin_config.yaml --wandb

# Evaluate models
echo "Evaluating models..."
python experiments/evaluate.py --checkpoint outputs/models/best_resnet18.pth
python experiments/evaluate.py --checkpoint outputs/models/best_swin_tiny.pth

# Generate XAI explanations
echo "Generating XAI explanations..."
python experiments/explain.py --checkpoint outputs/models/best_resnet18.pth --method all
python experiments/explain.py --checkpoint outputs/models/best_swin_tiny.pth --method all

echo "All experiments complete!"
