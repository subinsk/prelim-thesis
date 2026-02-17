"""
Test script to validate that all components are working correctly.

Runs quick tests on all modules before full training.

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Fix Windows encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import torch
        import torchvision
        import timm
        import yaml
        import cv2
        import matplotlib
        import seaborn
        import sklearn
        from pytorch_grad_cam import GradCAM
        print("  ✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_models():
    """Test that models can be instantiated."""
    print("\nTesting models...")

    try:
        from src.models.resnet import ResNet18Classifier
        from src.models.swin import get_swin_tiny
        from src.models.mil_aggregator import MILAggregator

        # Test ResNet18
        model_resnet = ResNet18Classifier(num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 256, 256)
        out = model_resnet(x)
        assert out.shape == (2, 2), f"ResNet output shape mismatch: {out.shape}"
        print("  ✓ ResNet18 working")

        # Test Swin (requires 224x224)
        model_swin = get_swin_tiny(pretrained=False, num_classes=2)
        x_swin = torch.randn(2, 3, 224, 224)
        out = model_swin(x_swin)
        assert out.shape == (2, 2), f"Swin output shape mismatch: {out.shape}"
        print("  ✓ Swin Transformer working")

        # Test MIL aggregator
        mil_agg = MILAggregator(aggregation='max', num_classes=2)
        patch_logits = torch.randn(4, 100, 2)  # batch=4, patches=100, classes=2
        agg_out = mil_agg(patch_logits)
        assert agg_out.shape == (4, 2), f"MIL output shape mismatch: {agg_out.shape}"
        print("  ✓ MIL Aggregator working")

        return True
    except Exception as e:
        print(f"  ✗ Model test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_pipeline():
    """Test data loading components."""
    print("\nTesting data pipeline...")

    try:
        from src.data.transforms import get_train_transforms, get_val_transforms

        # Test transforms
        train_transform = get_train_transforms(256)
        val_transform = get_val_transforms(256)

        from PIL import Image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        train_tensor = train_transform(dummy_img)
        assert train_tensor.shape == (3, 256, 256), f"Transform output shape mismatch: {train_tensor.shape}"
        print("  ✓ Transforms working")

        return True
    except Exception as e:
        print(f"  ✗ Data pipeline test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_xai():
    """Test XAI components."""
    print("\nTesting XAI...")

    try:
        from src.models.resnet import ResNet18Classifier
        from src.xai.gradcam import GradCAMExplainer

        model = ResNet18Classifier(num_classes=2, pretrained=False)
        model.eval()

        target_layers = [model.backbone.layer4[-1]]
        explainer = GradCAMExplainer(model, target_layers, device='cpu')

        x = torch.randn(1, 3, 256, 256)
        heatmap = explainer.explain(x, method='gradcam')

        assert isinstance(heatmap, np.ndarray), "Heatmap should be numpy array"
        assert heatmap.shape == (256, 256), f"Heatmap shape mismatch: {heatmap.shape}"
        assert 0 <= heatmap.min() <= heatmap.max() <= 1, "Heatmap values should be in [0, 1]"

        print("  ✓ GradCAM working")
        return True
    except Exception as e:
        print(f"  ✗ XAI test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")

    try:
        from src.utils.metrics import (
            compute_classification_metrics,
            compute_iou,
            compute_pointing_game
        )

        # Classification metrics
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.3, 0.2])

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        print("  ✓ Classification metrics working")

        # XAI metrics
        pred_mask = np.random.rand(100, 100)
        gt_mask = np.zeros((100, 100))
        gt_mask[20:50, 30:70] = 1

        iou = compute_iou(pred_mask, gt_mask)
        assert 0 <= iou <= 1, f"IoU should be in [0, 1], got {iou}"
        print("  ✓ XAI metrics working")

        return True
    except Exception as e:
        print(f"  ✗ Metrics test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization utilities."""
    print("\nTesting visualization...")

    try:
        from src.utils.visualization import (
            plot_confusion_matrix,
            overlay_heatmap
        )
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        # Test overlay
        image = np.random.rand(256, 256, 3)
        heatmap = np.random.rand(256, 256)

        overlay = overlay_heatmap(image, heatmap)
        assert overlay.shape == (256, 256, 3), f"Overlay shape mismatch: {overlay.shape}"

        print("  ✓ Visualization working")
        return True
    except Exception as e:
        print(f"  ✗ Visualization test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Test GPU inference
        try:
            model = torch.nn.Linear(10, 2).cuda()
            x = torch.randn(5, 10).cuda()
            out = model(x)
            print(f"  ✓ GPU inference working")
        except Exception as e:
            print(f"  ✗ GPU inference failed: {e}")
    else:
        print("  ⚠ CUDA not available - will use CPU (slower)")

    return True


def check_dataset():
    """Check if dataset is downloaded."""
    print("\nChecking dataset...")

    data_path = Path("data/cosoco")

    if not data_path.exists():
        print("  ✗ Dataset not found")
        print("    Run: python scripts/download_dataset.py")
        return False

    # Check splits
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            print(f"  ✗ {split} split not found")
            return False

        # Count files
        benign_count = len(list((split_path / 'benign').glob('*.png')))
        comp_count = len(list((split_path / 'compromised').glob('*.png')))

        # Exclude masks
        benign_count = len([f for f in (split_path / 'benign').glob('*.png') if '_mask' not in f.name])
        comp_count = len([f for f in (split_path / 'compromised').glob('*.png') if '_mask' not in f.name])

        print(f"  ✓ {split}: {benign_count} benign, {comp_count} compromised")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Mid-Semester Implementation Setup")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Models", test_models),
        ("Data Pipeline", test_data_pipeline),
        ("XAI", test_xai),
        ("Metrics", test_metrics),
        ("Visualization", test_visualization),
        ("GPU", check_gpu),
        ("Dataset", check_dataset),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s} {status}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready to run experiments.")
        print("\nNext steps:")
        print("  1. Download dataset: python scripts/download_dataset.py")
        print("  2. Train baseline: python experiments/train.py --model resnet18 --epochs 30")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before proceeding.")
        print("\nTroubleshooting:")
        print("  - Check requirements.txt: pip install -r requirements.txt")
        print("  - Verify Python version: python --version (need 3.9+)")
        if not results.get("Dataset", False):
            print("  - Download dataset: python scripts/download_dataset.py")

    print("=" * 60)


if __name__ == '__main__':
    main()
