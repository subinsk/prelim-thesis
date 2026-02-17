"""
Create dummy COSOCO dataset for quick testing.

This creates a small synthetic dataset to test the full pipeline
before the real dataset download completes.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def create_dummy_cosoco(output_dir="data/cosoco", samples_per_split=10):
    """
    Create dummy dataset with same structure as COSOCO.

    Args:
        output_dir: Output directory
        samples_per_split: Number of samples per split per class
    """
    output_path = Path(output_dir)
    print(f"Creating dummy COSOCO dataset at {output_path}")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['benign', 'compromised']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

    # Generate dummy images
    for split in ['train', 'val', 'test']:
        print(f"\nGenerating {split} split...")

        for class_name in ['benign', 'compromised']:
            class_dir = output_path / split / class_name
            label = 0 if class_name == 'benign' else 1

            for i in tqdm(range(samples_per_split), desc=f"  {class_name}"):
                # Create dummy image (1024x4096 as per COSOCO)
                # Use different patterns for benign vs compromised
                if label == 0:
                    # Benign: more uniform pattern
                    img_array = np.random.randint(100, 150, (1024, 4096, 3), dtype=np.uint8)
                else:
                    # Compromised: more varied pattern
                    img_array = np.random.randint(50, 200, (1024, 4096, 3), dtype=np.uint8)
                    # Add some "malicious" regions
                    img_array[200:400, 500:1000] = np.random.randint(200, 255, (200, 500, 3), dtype=np.uint8)

                # Save image
                img = Image.fromarray(img_array)
                img_path = class_dir / f"{class_name}_{i:05d}.png"
                img.save(img_path)

                # For compromised, create a dummy mask
                if label == 1:
                    mask = np.zeros((1024, 4096), dtype=np.uint8)
                    mask[200:400, 500:1000] = 255  # Mark malicious region
                    mask_img = Image.fromarray(mask)
                    mask_path = class_dir / f"{class_name}_{i:05d}_mask.png"
                    mask_img.save(mask_path)

    print(f"\n✓ Dummy dataset created!")
    print(f"\nDataset statistics:")
    for split in ['train', 'val', 'test']:
        benign_count = len(list((output_path / split / 'benign').glob('*.png')))
        comp_count = len(list((output_path / split / 'compromised').glob('*.png')))
        # Don't count masks
        benign_count = len([f for f in (output_path / split / 'benign').glob('*.png') if '_mask' not in f.name])
        comp_count = len([f for f in (output_path / split / 'compromised').glob('*.png') if '_mask' not in f.name])

        print(f"  {split}:")
        print(f"    Benign: {benign_count}")
        print(f"    Compromised: {comp_count}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10, help='Samples per split per class')
    args = parser.parse_args()

    create_dummy_cosoco(samples_per_split=args.samples)
