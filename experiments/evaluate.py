"""
Evaluation script for trained models.

Evaluates a trained model on the test set and computes detailed metrics
including accuracy, precision, recall, F1, confusion matrix, and ROC curves.

Usage:
    python experiments/evaluate.py --checkpoint outputs/models/best_resnet18.pth
    python experiments/evaluate.py --checkpoint outputs/models/swin_tiny.pth --test-only
"""

import argparse
import yaml
import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import COSOCODataset, collate_mil_batch
from src.data.transforms import get_val_transforms
from src.models.resnet import ResNet18Classifier
from src.models.swin import SwinTransformerClassifier, get_swin_tiny
from src.models.mil_aggregator import MILAggregator
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (overrides checkpoint config)')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to evaluate (train/val/test)')
    parser.add_argument('--output-dir', type=str, default='outputs/results',
                       help='Directory to save results')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to file')
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint['config']
    model_name = config['model']['name']

    # Initialize model
    if model_name == 'resnet18':
        model = ResNet18Classifier(
            num_classes=config['model']['num_classes'],
            pretrained=False,
            dropout=config['model']['dropout']
        )
    elif model_name == 'swin_tiny':
        model = get_swin_tiny(
            pretrained=False,
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        )
    elif 'swin' in model_name.lower():
        model = SwinTransformerClassifier(
            num_classes=config['model']['num_classes'],
            pretrained=False,
            model_name=model_name if 'patch4' in model_name else 'swin_tiny_patch4_window7_224',
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded {model_name} from checkpoint")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

    return model, config


def evaluate_model(model, test_loader, device, mil_aggregator):
    """
    Evaluate model and compute metrics.

    Returns:
        Dictionary with predictions, ground truth, probabilities, and metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_image_paths = []

    print("\nRunning evaluation...")

    with torch.no_grad():
        for batch_list in tqdm(test_loader, desc='Evaluating'):
            for sample in batch_list:
                # Get patches and label
                patches = sample['patches'].to(device)
                label = sample['image_label']
                image_path = sample['image_path']

                # Forward pass through all patches
                patch_logits = []
                for i in range(patches.size(0)):
                    patch = patches[i:i+1]
                    logits = model(patch)
                    patch_logits.append(logits)

                # Stack and aggregate
                patch_logits = torch.cat(patch_logits, dim=0).unsqueeze(0)
                image_logits = mil_aggregator(patch_logits)

                # Get prediction and probability
                probs = torch.softmax(image_logits, dim=1)
                pred = torch.argmax(image_logits, dim=1)

                all_preds.append(pred.cpu().item())
                all_labels.append(label)
                all_probs.append(probs[0, 1].cpu().item())  # Probability of class 1
                all_image_paths.append(image_path)

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'image_paths': all_image_paths,
        'metrics': metrics
    }


def save_results(results, output_dir, model_name):
    """Save evaluation results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_path = output_dir / f'{model_name}_metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in results['metrics'].items()
        }
        json.dump(metrics_serializable, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save predictions if requested
    predictions_path = output_dir / f'{model_name}_predictions.npz'
    np.savez(
        predictions_path,
        predictions=results['predictions'],
        labels=results['labels'],
        probabilities=results['probabilities']
    )
    print(f"Saved predictions to {predictions_path}")

    # Generate and save confusion matrix
    cm_path = output_dir / f'{model_name}_confusion_matrix.png'
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        class_names=['Benign', 'Compromised'],
        save_path=str(cm_path),
        normalize=False
    )

    # Generate and save ROC curve
    roc_path = output_dir / f'{model_name}_roc_curve.png'
    plot_roc_curve(
        results['labels'],
        results['probabilities'],
        save_path=str(roc_path)
    )

    return output_dir


def print_metrics(metrics, model_name):
    """Print metrics in a nice format."""
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results - {model_name}")
    print('=' * 60)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics.get('tn', 'N/A')}")
    print(f"  False Positives: {metrics.get('fp', 'N/A')}")
    print(f"  False Negatives: {metrics.get('fn', 'N/A')}")
    print(f"  True Positives:  {metrics.get('tp', 'N/A')}")

    # Compare to baseline
    baseline_f1 = 0.736
    print(f"\nComparison to Baseline (Paper):")
    print(f"  Target F1:  {baseline_f1:.4f}")
    print(f"  Our F1:     {metrics['f1']:.4f}")

    if metrics['f1'] >= baseline_f1:
        print(f"  Status:     ✓ ACHIEVED (Δ = +{metrics['f1'] - baseline_f1:.4f})")
    else:
        print(f"  Status:     ✗ Below target (Δ = {metrics['f1'] - baseline_f1:.4f})")

    print('=' * 60)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Override config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Load test dataset
    print(f"\nLoading {args.split} dataset...")
    test_dataset = COSOCODataset(
        data_dir=config['data']['dataset_path'],
        split=args.split,
        transform=get_val_transforms(config['data']['patch_size']),
        patch_size=config['data']['patch_size'],
        max_patches_per_image=config['data']['max_patches_per_image']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collate_mil_batch,
        pin_memory=config['hardware']['pin_memory']
    )

    # Initialize MIL aggregator
    mil_aggregator = MILAggregator(
        aggregation=config['mil']['aggregation'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # Run evaluation
    results = evaluate_model(model, test_loader, device, mil_aggregator)

    # Print results
    model_name = config['model']['name']
    print_metrics(results['metrics'], model_name)

    # Save results
    output_dir = save_results(results, args.output_dir, model_name)

    print(f"\n✓ Evaluation complete!")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
