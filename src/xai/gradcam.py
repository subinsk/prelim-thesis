"""
GradCAM and variants for explainability.

Implements Gradient-weighted Class Activation Mapping for explaining
model decisions on container malware images.

Classes:
    GradCAMExplainer: Main explainer class supporting multiple CAM variants
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2


class GradCAMExplainer:
    """
    GradCAM-based explainability for malware detection.

    Supports multiple CAM variants: GradCAM, HiResCAM, GradCAM++, EigenCAM.

    Args:
        model: PyTorch model
        target_layers: List of target layers for CAM extraction
        device: Device for computation ('cuda', 'cpu', 'mps')

    Example:
        >>> explainer = GradCAMExplainer(model, [model.layer4[-1]])
        >>> heatmap = explainer.explain(image_tensor, method='gradcam')
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        device: str = 'cuda'
    ):
        """Initialize GradCAM explainer."""
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.model.eval()

        # Initialize different CAM methods
        self.cam_methods = {
            'gradcam': GradCAM(model=model, target_layers=target_layers),
            'hirescam': HiResCAM(model=model, target_layers=target_layers),
            'gradcam++': GradCAMPlusPlus(model=model, target_layers=target_layers),
            'eigencam': EigenCAM(model=model, target_layers=target_layers)
        }

        print(f"Initialized GradCAMExplainer with {len(self.cam_methods)} methods")

    def explain(
        self,
        image_tensor: torch.Tensor,
        method: str = 'gradcam',
        target_class: Optional[int] = None,
        return_rgb: bool = False
    ) -> np.ndarray:
        """
        Generate CAM heatmap for an image.

        Args:
            image_tensor: Input image tensor [C, H, W] or [1, C, H, W]
            method: CAM method ('gradcam', 'hirescam', 'gradcam++', 'eigencam')
            target_class: Target class (None for predicted class)
            return_rgb: Return RGB overlay instead of grayscale heatmap

        Returns:
            heatmap: CAM heatmap [H, W] or RGB overlay [H, W, 3]
        """
        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        # Get CAM method
        if method not in self.cam_methods:
            raise ValueError(f"Unknown method {method}. Choose from {list(self.cam_methods.keys())}")

        cam = self.cam_methods[method]

        # Set target
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]

        # Generate CAM
        with torch.set_grad_enabled(True):
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

        # grayscale_cam shape: [batch_size, H, W]
        heatmap = grayscale_cam[0]  # Get first image in batch

        if return_rgb:
            # Convert image tensor to numpy for overlay
            img_np = self._tensor_to_numpy(image_tensor[0])
            # Create RGB overlay
            cam_image = show_cam_on_image(img_np, heatmap, use_rgb=True)
            return cam_image
        else:
            return heatmap

    def explain_patches(
        self,
        patches_tensor: torch.Tensor,
        method: str = 'gradcam',
        target_class: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate CAM heatmaps for multiple patches.

        Args:
            patches_tensor: Tensor of patches [num_patches, C, H, W]
            method: CAM method
            target_class: Target class

        Returns:
            List of heatmaps, one per patch
        """
        heatmaps = []

        for i in range(patches_tensor.size(0)):
            patch = patches_tensor[i:i+1]  # Keep batch dimension
            heatmap = self.explain(patch, method=method, target_class=target_class)
            heatmaps.append(heatmap)

        return heatmaps

    def aggregate_patch_explanations(
        self,
        patch_heatmaps: List[np.ndarray],
        patch_coords: List[tuple],
        image_size: tuple,
        aggregation: str = 'max'
    ) -> np.ndarray:
        """
        Aggregate patch-level heatmaps into full image heatmap.

        Args:
            patch_heatmaps: List of patch heatmaps
            patch_coords: List of (x, y) coordinates for each patch
            image_size: Original image size (H, W)
            aggregation: Aggregation method ('max', 'mean')

        Returns:
            Full image heatmap [H, W]
        """
        H, W = image_size
        full_heatmap = np.zeros((H, W))
        counts = np.zeros((H, W))  # Track overlaps

        patch_size = patch_heatmaps[0].shape[0]

        for heatmap, (x, y) in zip(patch_heatmaps, patch_coords):
            # Resize heatmap if needed
            if heatmap.shape[0] != patch_size:
                heatmap = cv2.resize(heatmap, (patch_size, patch_size))

            # Place in full image
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            h_end = y_end - y
            w_end = x_end - x

            if aggregation == 'max':
                full_heatmap[y:y_end, x:x_end] = np.maximum(
                    full_heatmap[y:y_end, x:x_end],
                    heatmap[:h_end, :w_end]
                )
            elif aggregation == 'mean':
                full_heatmap[y:y_end, x:x_end] += heatmap[:h_end, :w_end]
                counts[y:y_end, x:x_end] += 1

        if aggregation == 'mean':
            # Avoid division by zero
            counts = np.maximum(counts, 1)
            full_heatmap = full_heatmap / counts

        return full_heatmap

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert image tensor to numpy array for visualization."""
        # tensor: [C, H, W]
        img = tensor.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))  # [H, W, C]

        # Denormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean

        # Clip to [0, 1]
        img = np.clip(img, 0, 1)

        return img

    def get_top_k_patches(
        self,
        patch_heatmaps: List[np.ndarray],
        k: int = 10
    ) -> List[int]:
        """
        Get indices of top-k most salient patches.

        Args:
            patch_heatmaps: List of patch heatmaps
            k: Number of patches to return

        Returns:
            List of patch indices sorted by saliency
        """
        # Compute mean activation for each patch
        patch_scores = [heatmap.mean() for heatmap in patch_heatmaps]

        # Get top-k indices
        top_k_indices = np.argsort(patch_scores)[::-1][:k]

        return top_k_indices.tolist()

    def cleanup(self):
        """Release resources."""
        for cam in self.cam_methods.values():
            if hasattr(cam, '__del__'):
                cam.__del__()


def get_target_layer(model, model_name: str):
    """
    Get appropriate target layer for different model architectures.

    Args:
        model: PyTorch model
        model_name: Model architecture name

    Returns:
        List of target layers for GradCAM
    """
    if 'resnet' in model_name.lower():
        # Last conv layer in ResNet
        return [model.backbone.layer4[-1]]
    elif 'swin' in model_name.lower():
        # Last layer in Swin Transformer
        # Swin doesn't have traditional conv layers, use last block
        return [model.backbone.layers[-1].blocks[-1].norm1]
    elif 'vit' in model_name.lower():
        # Last attention block in ViT
        return [model.backbone.blocks[-1].norm1]
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


if __name__ == '__main__':
    # Test the explainer
    print("Testing GradCAM Explainer...")

    from src.models.resnet import ResNet18Classifier

    # Create dummy model
    model = ResNet18Classifier(num_classes=2, pretrained=False)
    model.eval()

    # Get target layer
    target_layers = [model.backbone.layer4[-1]]

    # Create explainer
    explainer = GradCAMExplainer(
        model=model,
        target_layers=target_layers,
        device='cpu'
    )

    # Create dummy input
    x = torch.randn(1, 3, 256, 256)

    # Generate explanation
    heatmap = explainer.explain(x, method='gradcam')

    print(f"Input shape: {x.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    print("\nGradCAM ready!")
