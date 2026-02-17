"""
Evaluation metrics for classification and XAI.

Contains functions for computing classification metrics (accuracy, F1, etc.)
and XAI metrics (IoU, Pointing Game, Faithfulness).

Functions:
    compute_classification_metrics: Standard classification metrics
    compute_iou: Intersection over Union for segmentation
    compute_pointing_game: Pointing game accuracy
    compute_faithfulness: Model faithfulness via perturbation
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional
import cv2


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary with accuracy, precision, recall, F1, and optionally AUC
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }

    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc_roc'] = 0.0

    # Add confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)

    return metrics


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union between predicted and ground-truth masks.

    Args:
        pred_mask: Predicted attention map (values in [0, 1])
        gt_mask: Ground truth binary mask
        threshold: Threshold for binarizing prediction

    Returns:
        IoU score (0 to 1)
    """
    # Binarize prediction
    pred_binary = (pred_mask >= threshold).astype(np.float32)

    # Binarize ground truth if needed
    gt_binary = (gt_mask > 0).astype(np.float32)

    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    iou = intersection / union
    return float(iou)


def compute_pointing_game(pred_mask: np.ndarray, gt_mask: np.ndarray) -> int:
    """
    Pointing Game: Does max attention fall within ground-truth region?

    Args:
        pred_mask: Predicted attention map
        gt_mask: Ground truth binary mask

    Returns:
        1 if hit (max point is in GT), 0 if miss
    """
    # Find location of maximum attention
    max_idx = np.unravel_index(np.argmax(pred_mask), pred_mask.shape)

    # Check if it falls within ground truth
    hit = int(gt_mask[max_idx] > 0)

    return hit


def compute_pointing_game_accuracy(pred_masks: list, gt_masks: list) -> float:
    """
    Compute Pointing Game accuracy over multiple samples.

    Args:
        pred_masks: List of predicted attention maps
        gt_masks: List of ground truth masks

    Returns:
        Pointing game accuracy (fraction of hits)
    """
    hits = 0
    total = len(pred_masks)

    for pred, gt in zip(pred_masks, gt_masks):
        hits += compute_pointing_game(pred, gt)

    return hits / total if total > 0 else 0.0


def compute_faithfulness(
    model,
    image: torch.Tensor,
    heatmap: np.ndarray,
    num_steps: int = 10,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Compute model faithfulness via iterative perturbation.

    Measures correlation between attention and model performance.
    Higher attention regions should have more impact when perturbed.

    Args:
        model: PyTorch model
        image: Input image tensor [C, H, W]
        heatmap: Attention heatmap [H, W]
        num_steps: Number of perturbation steps
        device: Device for computation

    Returns:
        (deletion_score, insertion_score)
        - deletion_score: AUC when removing high attention regions
        - insertion_score: AUC when adding high attention regions
    """
    model.eval()

    # Ensure image has batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    # Get original prediction
    with torch.no_grad():
        orig_output = model(image)
        orig_prob = torch.softmax(orig_output, dim=1)[0, 1].item()  # Prob of class 1

    # Resize heatmap to image size
    H, W = image.shape[2:]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Get sorted pixel indices by attention
    pixel_indices = np.argsort(heatmap_resized.flatten())[::-1]  # Descending order

    # Deletion: progressively remove high attention pixels
    deletion_probs = [orig_prob]
    perturbed_image = image.clone()

    pixels_per_step = len(pixel_indices) // num_steps

    for step in range(1, num_steps + 1):
        # Get pixels to perturb in this step
        end_idx = min(step * pixels_per_step, len(pixel_indices))
        indices_to_remove = pixel_indices[:end_idx]

        # Convert flat indices to 2D
        rows = indices_to_remove // W
        cols = indices_to_remove % W

        # Perturb by setting to zero (or mean)
        perturbed_image[0, :, rows, cols] = 0

        # Get new prediction
        with torch.no_grad():
            output = model(perturbed_image)
            prob = torch.softmax(output, dim=1)[0, 1].item()

        deletion_probs.append(prob)

    # Insertion: start with blank image, progressively add high attention pixels
    insertion_probs = [0.0]
    perturbed_image = torch.zeros_like(image)

    for step in range(1, num_steps + 1):
        end_idx = min(step * pixels_per_step, len(pixel_indices))
        indices_to_add = pixel_indices[:end_idx]

        rows = indices_to_add // W
        cols = indices_to_add % W

        # Add pixels from original image
        perturbed_image[0, :, rows, cols] = image[0, :, rows, cols]

        with torch.no_grad():
            output = model(perturbed_image)
            prob = torch.softmax(output, dim=1)[0, 1].item()

        insertion_probs.append(prob)

    # Compute AUC (area under curve)
    deletion_auc = np.trapz(deletion_probs, dx=1.0 / num_steps)
    insertion_auc = np.trapz(insertion_probs, dx=1.0 / num_steps)

    return deletion_auc, insertion_auc


def compute_average_precision(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Average Precision for attention map.

    Treats attention as "confidence scores" and GT as binary labels.

    Args:
        heatmap: Predicted attention map [H, W]
        gt_mask: Ground truth binary mask [H, W]

    Returns:
        Average Precision score
    """
    from sklearn.metrics import average_precision_score

    # Flatten
    heatmap_flat = heatmap.flatten()
    gt_flat = (gt_mask > 0).astype(int).flatten()

    # Compute AP
    ap = average_precision_score(gt_flat, heatmap_flat)

    return float(ap)


def compute_dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient (F1 for segmentation).

    Args:
        pred_mask: Predicted attention map
        gt_mask: Ground truth binary mask
        threshold: Threshold for binarizing prediction

    Returns:
        Dice coefficient (0 to 1)
    """
    # Binarize
    pred_binary = (pred_mask >= threshold).astype(np.float32)
    gt_binary = (gt_mask > 0).astype(np.float32)

    # Compute Dice
    intersection = (pred_binary * gt_binary).sum()
    dice = (2.0 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)

    return float(dice)


def evaluate_xai_method(
    pred_heatmaps: list,
    gt_masks: list,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Comprehensive XAI evaluation over multiple samples.

    Args:
        pred_heatmaps: List of predicted attention maps
        gt_masks: List of ground truth masks
        iou_threshold: Threshold for IoU computation

    Returns:
        Dictionary with mean IoU, Pointing Game accuracy, Dice, AP
    """
    ious = []
    dices = []
    aps = []

    for pred, gt in zip(pred_heatmaps, gt_masks):
        ious.append(compute_iou(pred, gt, iou_threshold))
        dices.append(compute_dice_coefficient(pred, gt, iou_threshold))
        aps.append(compute_average_precision(pred, gt))

    pg_acc = compute_pointing_game_accuracy(pred_heatmaps, gt_masks)

    return {
        'mean_iou': float(np.mean(ious)),
        'mean_dice': float(np.mean(dices)),
        'mean_ap': float(np.mean(aps)),
        'pointing_game_acc': float(pg_acc),
        'std_iou': float(np.std(ious)),
        'std_dice': float(np.std(dices))
    }


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics...")

    # Classification metrics
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.3, 0.2, 0.95, 0.15])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    print(f"\nClassification metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # XAI metrics
    pred_heatmap = np.random.rand(100, 100)
    gt_mask = np.zeros((100, 100))
    gt_mask[20:50, 30:70] = 1  # Ground truth region

    iou = compute_iou(pred_heatmap, gt_mask)
    pg = compute_pointing_game(pred_heatmap, gt_mask)
    dice = compute_dice_coefficient(pred_heatmap, gt_mask)

    print(f"\nXAI metrics:")
    print(f"  IoU: {iou:.3f}")
    print(f"  Pointing Game: {pg}")
    print(f"  Dice: {dice:.3f}")

    print("\nMetrics ready!")
