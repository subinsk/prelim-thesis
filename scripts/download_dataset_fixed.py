"""
Download COSOCO dataset directly to D drive (bypassing C drive cache).

Downloads the dataset using streaming to avoid C drive space issues.
"""

import os
from pathlib import Path

# CRITICAL: Set cache to D drive BEFORE importing anything from HuggingFace
# This MUST be done before importing datasets
os.environ['HF_HOME'] = 'D:\\projects\\prelim-thesis\\data\\.hf_cache'
os.environ['HF_DATASETS_CACHE'] = 'D:\\projects\\prelim-thesis\\data\\.hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:\\projects\\prelim-thesis\\data\\.hf_cache'
os.environ['HF_HUB_CACHE'] = 'D:\\projects\\prelim-thesis\\data\\.hf_cache'

# Now import datasets (it will use the cache location we set)
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_cosoco_to_d_drive():
    """
    Download COSOCO dataset with cache on D drive.
    """
    output_dir = Path('d:/projects/prelim-thesis/data/cosoco')
    cache_dir = Path('d:/projects/prelim-thesis/data/.hf_cache')

    print(f"Downloading to: {output_dir}")
    print(f"Cache location: {cache_dir}")
    print("This will take 30-45 minutes for ~20GB dataset (1024-unrolled config)\n")

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset with D drive cache
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset(
            'k3ylabs/cosoco-image-dataset',
            '1024-unrolled',  # Use 1024 config (20.4GB instead of 533GB)
            cache_dir=str(cache_dir)
        )

        print(f"\n✓ Dataset loaded!")
        print(f"Splits: {list(dataset.keys())}")

        # Save images to disk
        for split_name in dataset.keys():
            print(f"\n{'='*60}")
            print(f"Processing {split_name} split")
            print('='*60)

            split_data = dataset[split_name]

            # Create directories for each class
            for class_name in ['benign', 'compromised']:
                (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

            # Process each sample
            for idx, sample in enumerate(tqdm(split_data, desc=f"{split_name}")):
                image = sample['image']
                label = sample['label']

                # Determine class
                class_name = 'benign' if label == 0 else 'compromised'
                save_dir = output_dir / split_name / class_name

                # Save image
                image_path = save_dir / f"{class_name}_{idx:05d}.png"
                image.save(image_path)

                # Save mask if available
                if 'mask' in sample and sample['mask'] is not None:
                    mask_path = save_dir / f"{class_name}_{idx:05d}_mask.png"
                    sample['mask'].save(mask_path)

        print(f"\n{'='*60}")
        print("✓ Download complete!")
        print('='*60)

        # Print statistics
        print("\nDataset Statistics:")
        for split_name in ['train', 'val', 'test']:
            benign_imgs = [f for f in (output_dir / split_name / 'benign').glob('*.png') if '_mask' not in f.name]
            comp_imgs = [f for f in (output_dir / split_name / 'compromised').glob('*.png') if '_mask' not in f.name]

            print(f"  {split_name}:")
            print(f"    Benign: {len(benign_imgs)}")
            print(f"    Compromised: {len(comp_imgs)}")
            print(f"    Total: {len(benign_imgs) + len(comp_imgs)}")

        print(f"\n✓ All data saved to: {output_dir.absolute()}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check D drive has at least 5GB free space")
        print("2. Verify internet connection")
        print("3. Try running again (resume from where it stopped)")
        raise


if __name__ == '__main__':
    download_cosoco_to_d_drive()
