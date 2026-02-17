"""
Extract COSOCO dataset from tar files (bypassing buggy HuggingFace datasets library).

The tar files downloaded successfully, but load_dataset() has a schema bug.
This script extracts images directly from tar files.
"""

import tarfile
import json
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm


def extract_cosoco_from_cache():
    """
    Extract COSOCO images from cached tar files.
    """
    cache_dir = Path('d:/projects/prelim-thesis/data/.hf_cache')
    output_dir = Path('d:/projects/prelim-thesis/data/cosoco')

    # Find tar files
    tar_dir = cache_dir / 'datasets--k3ylabs--cosoco-image-dataset/snapshots'
    tar_files = list(tar_dir.glob('*/1024-unrolled/*.tar'))

    if not tar_files:
        print("❌ No tar files found in cache!")
        return

    print(f"Found {len(tar_files)} tar files")
    print(f"Extracting to: {output_dir}\n")

    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ['benign', 'compromised']:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    # Process each tar file
    stats = {'train': {'benign': 0, 'compromised': 0},
             'val': {'benign': 0, 'compromised': 0},
             'test': {'benign': 0, 'compromised': 0}}

    for tar_idx, tar_path in enumerate(sorted(tar_files), 1):
        # Determine split from filename
        if 'train' in tar_path.name:
            split = 'train'
        elif 'valid' in tar_path.name:
            split = 'val'
        elif 'test' in tar_path.name:
            split = 'test'
        else:
            continue

        print(f"\n[{tar_idx}/{len(tar_files)}] Processing {tar_path.name}...")

        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()

            # Group files by base name
            processed = set()

            for member in tqdm(members, desc=f"  {split}", leave=False):
                # Only process PNG images (not masks, not JSON)
                if not member.name.endswith('.png') or member.name.endswith('.mask.png'):
                    continue

                base_name = member.name.replace('.png', '')
                if base_name in processed:
                    continue
                processed.add(base_name)

                # Read JSON metadata to get label
                json_name = base_name + '.json'
                try:
                    json_member = tar.getmember(json_name)
                    json_file = tar.extractfile(json_member)
                    data = json.loads(json_file.read())
                    label = data.get('label', 'benign')  # Label is "benign" or "malevolent"
                except:
                    # Infer from filename if JSON fails
                    label = 'benign' if member.name.startswith('benign') else 'malevolent'

                # Map labels to class names
                class_name = 'benign' if label == 'benign' else 'compromised'

                # Extract and save image
                img_file = tar.extractfile(member)
                if img_file is None:
                    continue

                image = Image.open(io.BytesIO(img_file.read()))
                idx = stats[split][class_name]
                save_dir = output_dir / split / class_name
                image_path = save_dir / f"{class_name}_{idx:05d}.png"
                image.save(image_path)

                # Extract and save mask if available
                mask_name = base_name + '.mask.png'
                try:
                    mask_member = tar.getmember(mask_name)
                    mask_file = tar.extractfile(mask_member)
                    if mask_file:
                        mask = Image.open(io.BytesIO(mask_file.read()))
                        mask_path = save_dir / f"{class_name}_{idx:05d}_mask.png"
                        mask.save(mask_path)
                except:
                    pass  # No mask for this image

                stats[split][class_name] += 1

                # Log progress every 100 images
                total_processed = sum(sum(s.values()) for s in stats.values())
                if total_processed % 100 == 0:
                    print(f"  Progress: {total_processed} images extracted ({stats[split]['benign']} benign, {stats[split]['compromised']} compromised in {split})")

    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)

    print("\nDataset Statistics:")
    total_benign = 0
    total_compromised = 0
    for split in ['train', 'val', 'test']:
        benign = stats[split]['benign']
        comp = stats[split]['compromised']
        total_benign += benign
        total_compromised += comp
        print(f"  {split}:")
        print(f"    Benign: {benign}")
        print(f"    Compromised: {comp}")
        print(f"    Total: {benign + comp}")

    print(f"\nGrand Total:")
    print(f"  Benign: {total_benign}")
    print(f"  Compromised: {total_compromised}")
    print(f"  Total Images: {total_benign + total_compromised}")

    print(f"\n✓ All data saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    extract_cosoco_from_cache()
