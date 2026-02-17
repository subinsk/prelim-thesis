#!/bin/bash
# Script to download COSOCO dataset from HuggingFace

echo "Downloading COSOCO dataset from HuggingFace..."

# Create data directory
mkdir -p data

# Download using Python
python << EOF
from datasets import load_dataset
from huggingface_hub import snapshot_download

print("Loading COSOCO dataset...")
try:
    # Method 1: Using datasets library
    dataset = load_dataset("k3ylabs/cosoco-image-dataset")
    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Save to disk
    dataset.save_to_disk("data/cosoco")
    print("Dataset saved to data/cosoco/")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please check your internet connection and HuggingFace access")

EOF

echo "Download complete!"
