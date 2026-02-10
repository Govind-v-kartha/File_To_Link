"""
Image utilities for the Hybrid AI-Quantum Satellite Image Encryption System.
Handles image loading, validation, saving, and format conversions.
"""

import os
import hashlib
import numpy as np
from PIL import Image

from utils.logger import setup_logger, get_config_path

logger = setup_logger("IMAGE_UTILS", get_config_path())

SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "tiff", "tif"}


def validate_image(image_path: str) -> bool:
    """
    Validate that the image file exists, is a supported format, and can be opened.

    Args:
        image_path: Path to the image file.

    Returns:
        True if valid, raises exception otherwise.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: .{ext}. Supported: {SUPPORTED_FORMATS}"
        )

    try:
        with Image.open(image_path) as img:
            img.verify()
        logger.info(f"Image validated successfully: {image_path}")
        return True
    except Exception as e:
        raise ValueError(f"Image file is corrupted or invalid: {image_path}. Error: {e}")


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image as a NumPy RGB array (H, W, 3).

    Args:
        image_path: Path to the image file.

    Returns:
        NumPy array of shape (H, W, 3) with dtype uint8.
    """
    validate_image(image_path)
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    logger.info(
        f"Image loaded: {image_path}, shape={img_array.shape}, dtype={img_array.dtype}"
    )
    return img_array


def save_image(image_array: np.ndarray, output_path: str) -> str:
    """
    Save a NumPy array as an image file.

    Args:
        image_array: NumPy array of shape (H, W, 3) or (H, W), dtype uint8.
        output_path: Path where the image will be saved.

    Returns:
        The output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(output_path)
    logger.info(f"Image saved: {output_path}")
    return output_path


def compute_image_hash(image_array: np.ndarray) -> str:
    """
    Compute SHA-256 hash of an image array for integrity verification.

    Args:
        image_array: NumPy array of the image.

    Returns:
        SHA-256 hex digest string.
    """
    return hashlib.sha256(image_array.tobytes()).hexdigest()


def rgb_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using luminance weights.

    Args:
        image_array: NumPy array of shape (H, W, 3).

    Returns:
        NumPy array of shape (H, W) with dtype uint8.
    """
    if image_array.ndim == 2:
        return image_array
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = np.dot(image_array[..., :3], weights)
    return gray.astype(np.uint8)


def get_image_info(image_array: np.ndarray, filename: str = "") -> dict:
    """
    Get metadata information about an image.

    Args:
        image_array: NumPy array of the image.
        filename: Original filename.

    Returns:
        Dictionary with image metadata.
    """
    h, w = image_array.shape[:2]
    channels = image_array.shape[2] if image_array.ndim == 3 else 1
    return {
        "filename": filename,
        "size": [w, h],
        "channels": channels,
        "dtype": str(image_array.dtype),
        "hash": compute_image_hash(image_array),
    }


def list_input_images(input_dir: str) -> list:
    """
    List all supported image files in the input directory.

    Args:
        input_dir: Path to the input directory.

    Returns:
        List of full paths to image files.
    """
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory does not exist: {input_dir}")
        return []

    images = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower().lstrip(".")
        if ext in SUPPORTED_FORMATS:
            images.append(os.path.join(input_dir, fname))

    logger.info(f"Found {len(images)} image(s) in {input_dir}")
    return images
