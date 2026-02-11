"""
Block utilities for 32x32 pixel block division and reconstruction.
Handles ROI blocking, padding, block mapping, and reconstruction.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from utils.logger import setup_logger, get_config_path

logger = setup_logger("BLOCK_UTILS", get_config_path())

BLOCK_SIZE = 32


def create_roi_blocks(
    image: np.ndarray, roi_mask: np.ndarray
) -> Tuple[List[np.ndarray], List[dict], np.ndarray]:
    """
    Divide the ROI region of an image into 32x32 blocks for quantum encryption.

    This function extracts the bounding box of the ROI, then divides it into
    32x32 blocks in raster order. Partial blocks at the edges are zero-padded
    and padding information is stored for perfect reconstruction.

    Args:
        image: Original image array of shape (H, W, 3), dtype uint8.
        roi_mask: Binary mask of shape (H, W), 1 = ROI, 0 = background.

    Returns:
        Tuple of:
            - blocks: List of 8x8x3 numpy arrays (the blocks).
            - block_map: List of dicts with position/padding metadata per block.
            - roi_bbox: Array [y_min, x_min, y_max, x_max] of the ROI bounding box.
    """
    assert image.ndim == 3, f"Image must be 3D (H,W,C), got {image.ndim}D"
    assert roi_mask.ndim == 2, f"ROI mask must be 2D, got {roi_mask.ndim}D"
    assert image.shape[:2] == roi_mask.shape, (
        f"Image shape {image.shape[:2]} != mask shape {roi_mask.shape}"
    )

    # Ensure binary mask
    roi_mask_bin = (roi_mask > 0).astype(np.uint8)

    # Find bounding box of ROI
    rows = np.any(roi_mask_bin, axis=1)
    cols = np.any(roi_mask_bin, axis=0)

    if not rows.any():
        logger.warning("ROI mask is empty - no ROI pixels found")
        return [], [], np.array([0, 0, 0, 0])

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Include the max pixel
    y_max += 1
    x_max += 1

    roi_bbox = np.array([y_min, x_min, y_max, x_max])
    roi_region = image[y_min:y_max, x_min:x_max].copy()
    roi_h, roi_w = roi_region.shape[:2]

    logger.info(
        f"ROI bounding box: y=[{y_min},{y_max}], x=[{x_min},{x_max}], "
        f"size={roi_w}x{roi_h}"
    )

    # Calculate number of blocks in each dimension
    n_blocks_y = int(np.ceil(roi_h / BLOCK_SIZE))
    n_blocks_x = int(np.ceil(roi_w / BLOCK_SIZE))
    total_blocks = n_blocks_y * n_blocks_x

    logger.info(
        f"Dividing into {n_blocks_x}x{n_blocks_y} = {total_blocks} blocks of {BLOCK_SIZE}x{BLOCK_SIZE}"
    )

    blocks = []
    block_map = []
    padded_count = 0

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block_id = by * n_blocks_x + bx

            # Pixel coordinates within ROI region
            py_start = by * BLOCK_SIZE
            px_start = bx * BLOCK_SIZE
            py_end = min(py_start + BLOCK_SIZE, roi_h)
            px_end = min(px_start + BLOCK_SIZE, roi_w)

            # Extract the block
            raw_block = roi_region[py_start:py_end, px_start:px_end].copy()
            actual_h, actual_w = raw_block.shape[:2]

            # Calculate padding
            pad_bottom = BLOCK_SIZE - actual_h
            pad_right = BLOCK_SIZE - actual_w
            is_padded = pad_bottom > 0 or pad_right > 0

            if is_padded:
                padded_count += 1
                # Zero-pad to 8x8
                channels = raw_block.shape[2] if raw_block.ndim == 3 else 1
                if raw_block.ndim == 3:
                    padded_block = np.zeros(
                        (BLOCK_SIZE, BLOCK_SIZE, channels), dtype=raw_block.dtype
                    )
                else:
                    padded_block = np.zeros(
                        (BLOCK_SIZE, BLOCK_SIZE), dtype=raw_block.dtype
                    )
                padded_block[:actual_h, :actual_w] = raw_block
                block = padded_block
            else:
                block = raw_block

            # Global position (relative to original image)
            global_y = y_min + py_start
            global_x = x_min + px_start

            padding_info = None
            if is_padded:
                padding_info = {
                    "original_size": [actual_h, actual_w],
                    "padding": {"bottom": pad_bottom, "right": pad_right},
                }

            block_entry = {
                "block_id": block_id,
                "position": [global_x, global_y],
                "roi_local_position": [px_start, py_start],
                "size": [BLOCK_SIZE, BLOCK_SIZE],
                "is_padded": is_padded,
                "padding_info": padding_info,
            }

            blocks.append(block)
            block_map.append(block_entry)

    logger.info(
        f"Created {len(blocks)} blocks, {padded_count} blocks required padding"
    )

    return blocks, block_map, roi_bbox


def reconstruct_from_blocks(
    blocks: List[np.ndarray],
    block_map: List[dict],
    roi_bbox: np.ndarray,
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Reconstruct the ROI region from 8x8 blocks using the block map.

    Args:
        blocks: List of 8x8(x3) numpy arrays.
        block_map: List of dicts with position/padding metadata.
        roi_bbox: Array [y_min, x_min, y_max, x_max].
        original_shape: Shape of the original image (H, W, C).

    Returns:
        Reconstructed ROI image region of shape (roi_h, roi_w, C).
    """
    y_min, x_min, y_max, x_max = int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])
    roi_h = y_max - y_min
    roi_w = x_max - x_min
    channels = original_shape[2] if len(original_shape) > 2 else 1

    if channels > 1:
        roi_image = np.zeros((roi_h, roi_w, channels), dtype=np.uint8)
    else:
        roi_image = np.zeros((roi_h, roi_w), dtype=np.uint8)

    for block, bmap in zip(blocks, block_map):
        # Local position within the ROI region
        px_start = int(bmap["roi_local_position"][0])
        py_start = int(bmap["roi_local_position"][1])

        # Determine actual size (remove padding if needed)
        if bmap["is_padded"] and bmap["padding_info"]:
            actual_h = int(bmap["padding_info"]["original_size"][0])
            actual_w = int(bmap["padding_info"]["original_size"][1])
        else:
            actual_h = BLOCK_SIZE
            actual_w = BLOCK_SIZE

        # Place block (only the non-padded portion)
        block_region = block[:actual_h, :actual_w]

        # If output needs channels but block is grayscale, replicate to 3 channels
        if channels > 1 and block_region.ndim == 2:
            block_region = np.stack([block_region] * channels, axis=-1)

        roi_image[py_start : py_start + actual_h, px_start : px_start + actual_w] = (
            block_region
        )

    logger.info(f"Reconstructed ROI region: {roi_image.shape}")
    return roi_image


def place_roi_on_image(
    canvas: np.ndarray,
    roi_region: np.ndarray,
    roi_bbox: np.ndarray,
) -> np.ndarray:
    """
    Place a reconstructed ROI region back onto an image canvas.

    Args:
        canvas: The base image to place ROI onto.
        roi_region: The ROI region array.
        roi_bbox: Array [y_min, x_min, y_max, x_max].

    Returns:
        Updated canvas with ROI region placed.
    """
    y_min, x_min, y_max, x_max = int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])
    canvas[y_min:y_max, x_min:x_max] = roi_region
    return canvas


def blocks_to_flat_array(blocks: List[np.ndarray]) -> np.ndarray:
    """Stack all blocks into a single array for storage."""
    return np.stack(blocks, axis=0)


def flat_array_to_blocks(flat: np.ndarray) -> List[np.ndarray]:
    """Split a stacked array back into individual blocks."""
    return [flat[i] for i in range(flat.shape[0])]
