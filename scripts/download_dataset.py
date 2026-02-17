"""
Download COSOCO dataset from HuggingFace.

Downloads the COSOCO container malware image dataset and organizes it
into the expected directory structure.

Usage:
    python scripts/download_dataset.py
"""

import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm


def download_and_prepare_cosoco(output_dir="data/cosoco"):
    """
    Download COSOCO dataset and prepare directory structure.

    Expected structure:
        data/cosoco/
            train/
                benign/
                    image1.png
                    image2.png
                compromised/
                    image1.png
                    image1_mask.png
            val/
                benign/
                compromised/
            test/
                benign/
                compromised/
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading COSOCO dataset from HuggingFace...")
    print("This may take a while (dataset is ~4GB)...\n")

    try:
        # Load dataset from HuggingFace (using 4096-unrolled_n config)
        dataset = load_dataset('k3ylabs/cosoco-image-dataset', '4096-unrolled_n', cache_dir=str(output_path / '.cache'))

        print(f"Dataset loaded successfully!")
        print(f"Splits available: {list(dataset.keys())}")

        for split_name in dataset.keys():
            print(f"\nProcessing {split_name} split...")

            split_data = dataset[split_name]

            # Create split directory
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)

            # Create class directories
            benign_dir = split_dir / 'benign'
            compromised_dir = split_dir / 'compromised'
            benign_dir.mkdir(exist_ok=True)
            compromised_dir.mkdir(exist_ok=True)

            # Process each sample
            for idx, sample in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
                # Get image and label
                image = sample['image']  # PIL Image
                label = sample['label']  # 0 = benign, 1 = compromised

                # Determine output directory
                if label == 0:
                    save_dir = benign_dir
                    class_name = 'benign'
                else:
                    save_dir = compromised_dir
                    class_name = 'compromised'

                # Save image
                image_name = f"{class_name}_{idx:05d}.png"
                image_path = save_dir / image_name
                image.save(image_path)

                # Save mask if available
                if 'mask' in sample and sample['mask'] is not None:
                    mask = sample['mask']
                    mask_name = f"{class_name}_{idx:05d}_mask.png"
                    mask_path = save_dir / mask_name
                    mask.save(mask_path)

            print(f"  Saved {len(split_data)} images to {split_dir}")

        print(f"\n✓ Dataset downloaded and organized successfully!")
        print(f"Location: {output_path.absolute()}")

        # Print statistics
        print("\nDataset Statistics:")
        for split_name in dataset.keys():
            benign_count = len(list((output_path / split_name / 'benign').glob('*.png')))
            comp_count = len(list((output_path / split_name / 'compromised').glob('*.png')))
            # Don't count masks
            benign_count = len([f for f in (output_path / split_name / 'benign').glob('*.png') if '_mask' not in f.name])
            comp_count = len([f for f in (output_path / split_name / 'compromised').glob('*.png') if '_mask' not in f.name])

            print(f"  {split_name}:")
            print(f"    Benign: {benign_count}")
            print(f"    Compromised: {comp_count}")
            print(f"    Total: {benign_count + comp_count}")

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Install huggingface datasets: pip install datasets")
        print("3. Check HuggingFace dataset page: https://huggingface.co/datasets/k3ylabs/cosoco-image-dataset")
        raise


if __name__ == '__main__':
    download_and_prepare_cosoco()
