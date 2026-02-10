"""
Decision Engine - ROI Block Division.

Handles dividing the ROI region into 8x8 pixel blocks for quantum encryption.
Manages padding, block mapping, and provides reconstruction capabilities.
"""

import numpy as np
from typing import Tuple, List

from utils.logger import setup_logger, get_config_path
from utils.block_utils import (
    create_roi_blocks,
    reconstruct_from_blocks,
    place_roi_on_image,
    BLOCK_SIZE,
)

logger = setup_logger("DECISION_ENGINE", get_config_path())


def divide_roi_into_blocks(
    image: np.ndarray, roi_mask: np.ndarray
) -> Tuple[List[np.ndarray], List[dict], np.ndarray]:
    """
    Divide the ROI of an image into 8x8 blocks for quantum encryption.

    This is the main entry point for block division. It validates inputs,
    creates blocks, and logs comprehensive information about the result.

    Args:
        image: Original image (H, W, 3), dtype uint8.
        roi_mask: Binary ROI mask (H, W), 1 = ROI.

    Returns:
        Tuple of:
            - blocks: List of 8x8x3 NumPy arrays.
            - block_map: List of metadata dicts for each block.
            - roi_bbox: Array [y_min, x_min, y_max, x_max].
    """
    logger.info("=" * 60)
    logger.info("STARTING ROI BLOCK DIVISION")
    logger.info("=" * 60)

    # Validate inputs
    assert image.ndim == 3 and image.shape[2] == 3, (
        f"Expected 3-channel image, got shape {image.shape}"
    )
    assert roi_mask.ndim == 2, f"Expected 2D mask, got {roi_mask.ndim}D"
    assert image.shape[:2] == roi_mask.shape, (
        f"Image {image.shape[:2]} and mask {roi_mask.shape} size mismatch"
    )

    roi_pixel_count = np.sum(roi_mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    logger.info(
        f"Total ROI pixels: {roi_pixel_count} ({100 * roi_pixel_count / total_pixels:.1f}%)"
    )

    # Create blocks
    blocks, block_map, roi_bbox = create_roi_blocks(image, roi_mask)

    if len(blocks) == 0:
        logger.warning("No blocks created - ROI is empty")
        return blocks, block_map, roi_bbox

    # Log statistics
    padded_blocks = sum(1 for b in block_map if b["is_padded"])
    logger.info(
        f"Divided into {len(blocks)} {BLOCK_SIZE}x{BLOCK_SIZE} blocks, "
        f"{padded_blocks} blocks padded"
    )

    # Verify all blocks are 8x8
    for i, block in enumerate(blocks):
        assert block.shape[0] == BLOCK_SIZE and block.shape[1] == BLOCK_SIZE, (
            f"Block {i} has shape {block.shape}, expected ({BLOCK_SIZE}, {BLOCK_SIZE}, ...)"
        )

    logger.info(f"All {len(blocks)} blocks verified as {BLOCK_SIZE}x{BLOCK_SIZE}")

    # Verify reconstruction
    _verify_block_reconstruction(image, blocks, block_map, roi_bbox)

    logger.info("=" * 60)
    return blocks, block_map, roi_bbox


def _verify_block_reconstruction(
    original_image: np.ndarray,
    blocks: List[np.ndarray],
    block_map: List[dict],
    roi_bbox: np.ndarray,
) -> None:
    """
    Verify that blocks can perfectly reconstruct the ROI region.

    Args:
        original_image: Original image.
        blocks: List of 8x8 blocks.
        block_map: Block metadata.
        roi_bbox: ROI bounding box.
    """
    y_min, x_min, y_max, x_max = roi_bbox
    original_roi = original_image[y_min:y_max, x_min:x_max].copy()

    reconstructed_roi = reconstruct_from_blocks(
        blocks, block_map, roi_bbox, original_image.shape
    )

    diff = np.abs(
        original_roi.astype(np.int16) - reconstructed_roi.astype(np.int16)
    )
    max_diff = np.max(diff)

    if max_diff == 0:
        logger.info(
            "Block reconstruction verification PASSED: zero data loss in blocking"
        )
    else:
        logger.error(
            f"Block reconstruction verification FAILED: max pixel difference = {max_diff}"
        )
        raise RuntimeError(
            f"Block division introduces data loss! Max difference: {max_diff}"
        )


def reconstruct_roi_from_blocks(
    blocks: List[np.ndarray],
    block_map: List[dict],
    roi_bbox: np.ndarray,
    original_shape: tuple,
) -> np.ndarray:
    """
    Reconstruct the ROI region from decrypted blocks.

    Args:
        blocks: List of decrypted 8x8 blocks.
        block_map: Block metadata with positions and padding info.
        roi_bbox: ROI bounding box.
        original_shape: Shape of the original image.

    Returns:
        Reconstructed ROI region.
    """
    logger.info(f"Reconstructing ROI from {len(blocks)} blocks")
    roi_region = reconstruct_from_blocks(blocks, block_map, roi_bbox, original_shape)
    logger.info(f"ROI reconstructed: {roi_region.shape}")
    return roi_region


def get_block_statistics(block_map: List[dict]) -> dict:
    """
    Get statistics about the block division.

    Args:
        block_map: List of block metadata dicts.

    Returns:
        Dictionary with block statistics.
    """
    total = len(block_map)
    padded = sum(1 for b in block_map if b["is_padded"])

    return {
        "total_blocks": total,
        "padded_blocks": padded,
        "full_blocks": total - padded,
        "block_size": BLOCK_SIZE,
        "percent_padded": 100 * padded / total if total > 0 else 0,
    }
