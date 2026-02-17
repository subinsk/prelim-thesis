"""
Visualization utilities for results and explanations.

Functions for plotting training curves, confusion matrices, XAI heatmaps, etc.

Functions:
    plot_training_curves: Plot loss and accuracy over epochs
    plot_confusion_matrix: Visualize confusion matrix
    plot_gradcam_comparison: Compare multiple XAI methods
    plot_heatmap_overlay: Overlay heatmap on original image
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from typing import Optional, List, Dict
from pathlib import Path


# Set publication-quality style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.

    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Training accuracy per epoch
        val_accs: Validation accuracy per epoch
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
        normalize: Whether to normalize counts
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()
    plt.close()


def plot_gradcam_comparison(
    image: np.ndarray,
    gradcam_maps: Dict[str, np.ndarray],
    title: str = "GradCAM Comparison",
    save_path: Optional[str] = None
):
    """
    Plot side-by-side comparison of different GradCAM methods.

    Args:
        image: Original image [H, W, C] in range [0, 1]
        gradcam_maps: Dict of {method_name: heatmap}
        title: Figure title
        save_path: Path to save figure
    """
    num_methods = len(gradcam_maps)
    fig, axes = plt.subplots(1, num_methods + 1, figsize=(4 * (num_methods + 1), 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # GradCAM overlays
    for idx, (method, heatmap) in enumerate(gradcam_maps.items(), start=1):
        overlay = overlay_heatmap(image, heatmap)
        axes[idx].imshow(overlay)
        axes[idx].set_title(method)
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    plt.show()
    plt.close()


def plot_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet',
    save_path: Optional[str] = None
):
    """
    Overlay heatmap on original image.

    Args:
        image: Original image [H, W, C] or [H, W]
        heatmap: Attention heatmap [H, W]
        alpha: Transparency of overlay
        colormap: Matplotlib colormap name
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    # Show original image
    plt.subplot(1, 3, 1)
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Show heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap=colormap)
    plt.colorbar()
    plt.title('Attention Heatmap')
    plt.axis('off')

    # Show overlay
    plt.subplot(1, 3, 3)
    overlay = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved overlay to {save_path}")

    plt.show()
    plt.close()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Create overlay of heatmap on image.

    Args:
        image: Original image [H, W, C] in range [0, 1]
        heatmap: Attention map [H, W] in range [0, 1]
        alpha: Transparency
        colormap: Colormap name

    Returns:
        Overlaid image [H, W, C]
    """
    # Resize heatmap if needed
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap to heatmap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0

    # Handle grayscale images
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Blend
    overlay = (1 - alpha) * image + alpha * heatmap_colored

    return np.clip(overlay, 0, 1)


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot bar chart comparing metrics across methods.

    Args:
        results: Dict of {method_name: {metric_name: value}}
        metric_names: Which metrics to plot (None = all)
        save_path: Path to save figure
    """
    if metric_names is None:
        # Get all unique metric names
        metric_names = list(next(iter(results.values())).keys())

    num_metrics = len(metric_names)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))

    if num_metrics == 1:
        axes = [axes]

    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.6

    for idx, metric in enumerate(metric_names):
        values = [results[method].get(metric, 0) for method in methods]

        axes[idx].bar(x, values, width, alpha=0.8)
        axes[idx].set_xlabel('Method')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(methods, rotation=45, ha='right')
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")

    plt.show()
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    plt.show()
    plt.close()


def plot_patch_attention(
    patches: np.ndarray,
    attention_scores: np.ndarray,
    top_k: int = 9,
    save_path: Optional[str] = None
):
    """
    Plot top-k patches by attention score.

    Args:
        patches: Array of patches [num_patches, H, W, C]
        attention_scores: Attention score per patch [num_patches]
        top_k: Number of top patches to show
        save_path: Path to save figure
    """
    # Get top-k indices
    top_indices = np.argsort(attention_scores)[::-1][:top_k]

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(top_k)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for idx, patch_idx in enumerate(top_indices):
        patch = patches[patch_idx]
        score = attention_scores[patch_idx]

        axes[idx].imshow(patch)
        axes[idx].set_title(f'Rank {idx+1}\nScore: {score:.3f}', fontsize=9)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(top_indices), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Top-{top_k} Patches by Attention', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved patch attention to {save_path}")

    plt.show()
    plt.close()


def save_figure_publication_quality(fig, save_path: str, formats: List[str] = ['png', 'pdf']):
    """
    Save figure in publication quality.

    Args:
        fig: Matplotlib figure
        save_path: Base path (without extension)
        formats: List of formats to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = save_path.with_suffix(f'.{fmt}')
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")


if __name__ == '__main__':
    # Test visualizations
    print("Testing visualization utilities...")

    # Test training curves
    epochs = 10
    train_losses = np.exp(-np.linspace(0, 2, epochs)) + np.random.rand(epochs) * 0.1
    val_losses = np.exp(-np.linspace(0, 1.8, epochs)) + np.random.rand(epochs) * 0.15
    train_accs = 1 - np.exp(-np.linspace(0, 2.5, epochs)) * 0.5
    val_accs = 1 - np.exp(-np.linspace(0, 2.2, epochs)) * 0.5

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # Test confusion matrix
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])

    plot_confusion_matrix(y_true, y_pred, class_names=['Benign', 'Malicious'])

    print("\nVisualization utilities ready!")
