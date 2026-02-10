import numpy as np
from PIL import Image
import os

def tar_to_grayscale_image(tar_path, output_path, width=256):
    # Read tar file as raw bytes
    with open(tar_path, "rb") as f:
        byte_data = f.read()

    # Convert bytes to numpy array (0–255)
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)

    # Compute height based on fixed width
    height = int(np.ceil(len(byte_array) / width))

    # Pad if needed
    padded_length = width * height
    padded_array = np.pad(byte_array, (0, padded_length - len(byte_array)))

    # Reshape to 2D image
    image_array = padded_array.reshape((height, width))

    # Convert to image and save
    image = Image.fromarray(image_array, mode="L")
    image.save(output_path)

    print(f"Saved image to {output_path}")
    print(f"Image shape: {image_array.shape}")
