"""
Training script for container malware detection models.

This script trains CNN or Vision Transformer models on the COSOCO dataset
using Multiple Instance Learning (MIL) aggregation.

Usage:
    python experiments/train.py --config configs/config.yaml --model resnet18
    python experiments/train.py --config configs/swin_config.yaml --epochs 50
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import COSOCODataset, collate_mil_batch
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.resnet import ResNet18Classifier
from src.models.swin import SwinTransformerClassifier, get_swin_tiny
from src.models.mil_aggregator import MILAggregator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train malware detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup computation device."""
    device_name = config['hardware']['device']

    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def get_model(config, device):
    """Initialize model based on config."""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    dropout = config['model']['dropout']

    if model_name == 'resnet18':
        model = ResNet18Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name == 'swin_tiny':
        model = get_swin_tiny(
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout
        )
    elif 'swin' in model_name.lower():
        # Generic Swin support
        model = SwinTransformerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_name=model_name if 'patch4' in model_name else 'swin_tiny_patch4_window7_224',
            dropout=dropout
        )
    else:
        raise ValueError(f"Model {model_name} not yet implemented. Supported: resnet18, swin_tiny")

    return model.to(device)


def get_optimizer(model, config):
    """Initialize optimizer."""
    opt_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer


def get_scheduler(optimizer, config):
    """Initialize learning rate scheduler."""
    scheduler_name = config['training']['scheduler'].lower()
    epochs = config['training']['epochs']
    warmup_epochs = config['training']['warmup_epochs']

    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-6
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def compute_metrics(all_preds, all_labels):
    """Compute classification metrics."""
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / total

    # Per-class metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


def train_epoch(model, train_loader, criterion, optimizer, device, mil_aggregator, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc='Training')

    for batch_idx, batch_list in enumerate(pbar):
        # batch_list is a list of dicts (one per image)
        batch_loss = 0

        for sample in batch_list:
            # Get patches and label
            patches = sample['patches'].to(device)  # [num_patches, C, H, W]
            label = torch.tensor([sample['image_label']], dtype=torch.long).to(device)

            # Forward pass through all patches
            patch_logits = []
            for i in range(patches.size(0)):
                patch = patches[i:i+1]  # [1, C, H, W]
                logits = model(patch)  # [1, num_classes]
                patch_logits.append(logits)

            # Stack patch logits
            patch_logits = torch.cat(patch_logits, dim=0)  # [num_patches, num_classes]
            patch_logits = patch_logits.unsqueeze(0)  # [1, num_patches, num_classes]

            # MIL aggregation
            image_logits = mil_aggregator(patch_logits)  # [1, num_classes]

            # Compute loss
            loss = criterion(image_logits, label)
            batch_loss += loss

            # Get prediction
            pred = torch.argmax(image_logits, dim=1)
            all_preds.append(pred.cpu().item())
            all_labels.append(label.cpu().item())

        # Average loss over batch
        batch_loss = batch_loss / len(batch_list)

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])

        optimizer.step()

        total_loss += batch_loss.item()

        # Update progress bar
        pbar.set_postfix({'loss': f"{batch_loss.item():.4f}"})

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(train_loader)

    return metrics


def validate(model, val_loader, criterion, device, mil_aggregator):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_list in tqdm(val_loader, desc='Validating'):
            batch_loss = 0

            for sample in batch_list:
                patches = sample['patches'].to(device)
                label = torch.tensor([sample['image_label']], dtype=torch.long).to(device)

                # Forward pass
                patch_logits = []
                for i in range(patches.size(0)):
                    patch = patches[i:i+1]
                    logits = model(patch)
                    patch_logits.append(logits)

                patch_logits = torch.cat(patch_logits, dim=0).unsqueeze(0)
                image_logits = mil_aggregator(patch_logits)

                # Compute loss
                loss = criterion(image_logits, label)
                batch_loss += loss

                # Get prediction
                pred = torch.argmax(image_logits, dim=1)
                all_preds.append(pred.cpu().item())
                all_labels.append(label.cpu().item())

            batch_loss = batch_loss / len(batch_list)
            total_loss += batch_loss.item()

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(val_loader)

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }

    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / f"{config['model']['name']}_latest.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / f"{config['model']['name']}_best.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved BEST checkpoint to {best_path}")


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)

    # Override config with command line args
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup device
    device = setup_device(config)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = COSOCODataset(
        data_dir=config['data']['dataset_path'],
        split='train',
        transform=get_train_transforms(config['data']['patch_size']),
        patch_size=config['data']['patch_size'],
        max_patches_per_image=config['data']['max_patches_per_image']
    )

    val_dataset = COSOCODataset(
        data_dir=config['data']['dataset_path'],
        split='val',
        transform=get_val_transforms(config['data']['patch_size']),
        patch_size=config['data']['patch_size'],
        max_patches_per_image=config['data']['max_patches_per_image']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collate_mil_batch,
        pin_memory=config['hardware']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collate_mil_batch,
        pin_memory=config['hardware']['pin_memory']
    )

    # Initialize model
    print(f"\nInitializing {config['model']['name']}...")
    model = get_model(config, device)

    # Initialize MIL aggregator
    mil_aggregator = MILAggregator(
        aggregation=config['mil']['aggregation'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['metrics'].get('f1', 0.0)

    # Initialize wandb if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging']['wandb_entity'],
                config=config,
                name=f"{config['model']['name']}_{config['experiment_name']}"
            )
        except ImportError:
            print("Warning: wandb not installed, skipping logging")

    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs")
    print(f"Target F1 score: 0.736 (baseline from paper)\n")

    patience_counter = 0

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 50)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, mil_aggregator, config)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, mil_aggregator)

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")

        # Log to wandb
        if args.wandb:
            try:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'lr': optimizer.param_groups[0]['lr']
                })
            except:
                pass

        # Learning rate scheduling
        if scheduler:
            scheduler.step()

        # Check if best model
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            print(f"*** New best F1: {best_f1:.4f} ***")
        else:
            patience_counter += 1

        # Save checkpoint
        if (epoch + 1) % config['logging']['save_interval'] == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, is_best)

        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best F1: {best_f1:.4f}")
            break

    print("\n" + "=" * 50)
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Target was 0.736 (baseline from paper)")

    if best_f1 >= 0.736:
        print("✓ Target achieved!")
    else:
        print(f"Need to improve by {0.736 - best_f1:.4f}")

    if args.wandb:
        try:
            wandb.finish()
        except:
            pass


if __name__ == '__main__':
    main()
