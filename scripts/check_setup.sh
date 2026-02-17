#!/bin/bash
# Quick check if everything is set up correctly

echo "🔍 Checking thesis setup..."
echo ""

# Check Python
echo "1. Python version:"
python --version || echo "❌ Python not found"
echo ""

# Check virtual environment
echo "2. Virtual environment:"
if [ -d "venv" ]; then
    echo "✓ venv/ exists"
else
    echo "❌ venv/ not found - run: python -m venv venv"
fi
echo ""

# Check if venv is activated
echo "3. Environment active:"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment is active"
else
    echo "⚠ Virtual environment not activated"
    echo "   Run: source venv/bin/activate (or venv\Scripts\activate on Windows)"
fi
echo ""

# Check key packages
echo "4. Key packages:"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch not installed"
python -c "import torchvision; print(f'✓ torchvision')" 2>/dev/null || echo "❌ torchvision not installed"
python -c "import timm; print(f'✓ timm')" 2>/dev/null || echo "❌ timm not installed"
python -c "from pytorch_grad_cam import GradCAM; print(f'✓ pytorch-grad-cam')" 2>/dev/null || echo "❌ pytorch-grad-cam not installed"
echo ""

# Check CUDA
echo "5. GPU/CUDA:"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}');" 2>/dev/null
echo ""

# Check directory structure
echo "6. Directory structure:"
for dir in src/data src/models src/xai src/hilbert experiments notebooks configs outputs; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "❌ $dir/ missing"
    fi
done
echo ""

# Check dataset
echo "7. Dataset:"
if [ -d "data/cosoco" ]; then
    echo "✓ COSOCO dataset found"
else
    echo "❌ Dataset not downloaded"
    echo "   Run: bash scripts/download_data.sh"
fi
echo ""

echo "📋 Summary:"
echo "If any ❌ errors above, fix them before starting Week 3"
echo ""
