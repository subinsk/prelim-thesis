"""
Multiple Instance Learning aggregation for patch-level predictions.
"""

import torch
import torch.nn as nn


class MILAggregator(nn.Module):
    """
    Aggregate patch-level logits to image-level prediction.
    
    Supports: max, mean, noisy_or pooling.
    """
    
    def __init__(self, aggregation: str = 'max', num_classes: int = 2):
        super().__init__()
        self.aggregation = aggregation
        self.num_classes = num_classes
    
    def forward(self, patch_logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate patch logits to image logits.
        
        Args:
            patch_logits: [B, num_patches, num_classes] or [total_patches, num_classes]
            
        Returns:
            image_logits: [B, num_classes]
        """
        if self.aggregation == 'max':
            # Max pooling - if any patch is malicious, image is malicious
            if len(patch_logits.shape) == 3:
                return torch.max(patch_logits, dim=1)[0]
            else:
                # Assume single image
                return torch.max(patch_logits, dim=0, keepdim=True)[0]
        
        elif self.aggregation == 'mean':
            # Average pooling
            if len(patch_logits.shape) == 3:
                return torch.mean(patch_logits, dim=1)
            else:
                return torch.mean(patch_logits, dim=0, keepdim=True)
        
        elif self.aggregation == 'noisy_or':
            # Noisy-OR: P(positive) = 1 - prod(1 - p_i)
            probs = torch.softmax(patch_logits, dim=-1)
            
            if len(patch_logits.shape) == 3:
                # Batch processing
                mal_probs = probs[:, :, 1]  # P(malicious) for each patch
                combined = 1 - torch.prod(1 - mal_probs, dim=1)
                return torch.stack([1 - combined, combined], dim=1)
            else:
                mal_probs = probs[:, 1]
                combined = 1 - torch.prod(1 - mal_probs)
                return torch.stack([1 - combined, combined]).unsqueeze(0)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
