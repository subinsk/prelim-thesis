"""
Swin Transformer implementation for container malware detection.

Swin Transformer with hierarchical architecture for patch-level
classification. Uses timm library for pretrained weights.

Classes:
    SwinTransformerClassifier: Swin Transformer for classification

Functions:
    get_swin_tiny: Create Swin-Tiny model
    get_swin_small: Create Swin-Small model
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, List, Tuple


class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer for patch-level malware classification.

    Uses hierarchical vision transformer architecture with shifted windows.
    Pretrained on ImageNet-1K or ImageNet-21K.

    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use pretrained weights (default: True)
        model_name (str): Timm model name (default: 'swin_tiny_patch4_window7_224')
        dropout (float): Dropout rate (default: 0.1)

    Example:
        >>> model = SwinTransformerClassifier(num_classes=2)
        >>> output = model(patches)  # patches: [B, 3, 224, 224]
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = 'swin_tiny_patch4_window7_224',
        dropout: float = 0.1
    ):
        """Initialize Swin Transformer classifier."""
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained Swin Transformer from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )

        print(f"Initialized {model_name} with {self.feature_dim} features")
        if pretrained:
            print("Using ImageNet pretrained weights")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Swin Transformer.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            logits: Class logits [B, num_classes]
        """
        # Extract features using backbone
        features = self.backbone(x)  # [B, feature_dim]

        # Classification
        logits = self.classifier(features)  # [B, num_classes]

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings without classification.

        Used for visualization and feature analysis.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            features: Feature embeddings [B, feature_dim]
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention weights from all layers.

        Used for attention rollout and visualization.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            List of attention weight tensors from each layer
        """
        attention_weights = []

        def hook_fn(module, input, output):
            # Extract attention weights if available
            if hasattr(output, 'attn_weights'):
                attention_weights.append(output.attn_weights)

        # Register hooks on attention modules
        hooks = []
        for name, module in self.backbone.named_modules():
            if 'attn' in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            _ = self.backbone(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return attention_weights

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning only the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen - only training classifier head")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - training full model")


def get_swin_tiny(pretrained: bool = True, num_classes: int = 2, dropout: float = 0.1):
    """
    Factory function for Swin-Tiny model.

    Swin-Tiny: 28M parameters, suitable for most tasks.

    Args:
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate

    Returns:
        SwinTransformerClassifier instance
    """
    return SwinTransformerClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name='swin_tiny_patch4_window7_224',
        dropout=dropout
    )


def get_swin_small(pretrained: bool = True, num_classes: int = 2, dropout: float = 0.1):
    """
    Factory function for Swin-Small model.

    Swin-Small: 50M parameters, better accuracy but slower.

    Args:
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate

    Returns:
        SwinTransformerClassifier instance
    """
    return SwinTransformerClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name='swin_small_patch4_window7_224',
        dropout=dropout
    )


def get_swin_base(pretrained: bool = True, num_classes: int = 2, dropout: float = 0.1):
    """
    Factory function for Swin-Base model.

    Swin-Base: 88M parameters, highest accuracy.

    Args:
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate

    Returns:
        SwinTransformerClassifier instance
    """
    return SwinTransformerClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name='swin_base_patch4_window7_224',
        dropout=dropout
    )


if __name__ == '__main__':
    # Test the model
    print("Testing Swin Transformer...")

    model = get_swin_tiny(pretrained=False, num_classes=2)

    # Create dummy input
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Feature extraction
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")

    print("\nModel ready!")
