#!/bin/bash
# Setup script for thesis environment

echo "Setting up MTech Thesis environment..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/Scripts/activate" ]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating output directories..."
mkdir -p outputs/models
mkdir -p outputs/figures
mkdir -p outputs/results
mkdir -p outputs/explanations
mkdir -p logs

# Setup PYTHONPATH
echo "Setting up PYTHONPATH..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "Setup complete!"
echo "Activate the environment with: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)"
