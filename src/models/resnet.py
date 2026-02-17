"""
ResNet18 for patch-level container malware classification.

Uses ImageNet pretrained weights, replaces final FC for binary classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNet18Classifier(nn.Module):
    """
    ResNet18 for binary malware classification.
    
    Args:
        num_classes: Number of output classes (default: 2)
        pretrained: Use ImageNet weights (default: True)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [B, 3, H, W], Output: [B, num_classes]"""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features before final FC layer (for XAI).
        
        Returns: [B, 512, H', W'] features
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        return x


def get_resnet18(pretrained: bool = True, num_classes: int = 2) -> ResNet18Classifier:
    """Factory function to create ResNet18 model."""
    return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)


def get_target_layer_for_gradcam(model: ResNet18Classifier):
    """Get target layer for GradCAM (last conv layer)."""
    return [model.backbone.layer4[-1]]
