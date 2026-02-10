"""
Analyze Workflow - ROI Analysis Only.

Runs only the FlexiMo segmentation and block division steps without encryption.
Useful for previewing what regions will be quantum-encrypted.
"""

import os
import time
import numpy as np
from datetime import datetime

from utils.logger import setup_logger, get_config_path, load_config
from utils.image_utils import load_image, get_image_info
from engines.ai_engine import segment_image_fleximo, save_segmentation_visualization
from engines.decision_engine import divide_roi_into_blocks, get_block_statistics

logger = setup_logger("ANALYZE_WORKFLOW", get_config_path())


def run_analysis(
    image_path: str,
    output_dir: str = None,
    config: dict = None,
) -> dict:
    """
    Run ROI analysis (segmentation + block division) without encryption.

    Args:
        image_path: Path to the input satellite image.
        output_dir: Output directory. If None, uses config default.
        config: Configuration dict.

    Returns:
        Dictionary with analysis results.
    """
    start_time = time.time()

    if config is None:
        config = load_config()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(project_root, config["paths"]["output_dir"])

    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    image_basename = os.path.splitext(os.path.basename(image_path))[0]

    logger.info("=" * 70)
    logger.info("ROI ANALYSIS MODE")
    logger.info("=" * 70)
    logger.info(f"Input image: {image_path}")

    # Load image
    image = load_image(image_path)
    image_info = get_image_info(image, os.path.basename(image_path))
    logger.info(f"Image: {image.shape[1]}x{image.shape[0]}, {image.shape[2]} channels")

    # Run FlexiMo segmentation
    logger.info("\nRunning FlexiMo AI segmentation...")
    roi_mask, background_mask, seg_raw = segment_image_fleximo(image, config)

    # Save visualizations
    saved_files = save_segmentation_visualization(
        image, roi_mask, background_mask, analysis_dir, image_basename
    )

    # Run block division analysis
    logger.info("\nAnalyzing block division...")
    blocks, block_map, roi_bbox = divide_roi_into_blocks(image, roi_mask)
    block_stats = get_block_statistics(block_map)

    # Estimate encryption time
    shots = config.get("quantum_encryption", {}).get("shots", 1024)
    est_time_per_block = 1.5  # seconds (typical for NEQR on 8x8)
    est_total_time = block_stats["total_blocks"] * est_time_per_block

    elapsed = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS RESULTS")
    logger.info("=" * 70)
    logger.info(f"Image size: {image.shape[1]}x{image.shape[0]}")
    logger.info(f"Total pixels: {image.shape[0] * image.shape[1]}")
    logger.info(f"ROI pixels: {np.sum(roi_mask)} ({100 * np.sum(roi_mask) / (image.shape[0] * image.shape[1]):.1f}%)")
    logger.info(f"Background pixels: {np.sum(background_mask)}")
    logger.info(f"Total 8x8 blocks: {block_stats['total_blocks']}")
    logger.info(f"  Full blocks: {block_stats['full_blocks']}")
    logger.info(f"  Padded blocks: {block_stats['padded_blocks']}")
    logger.info(f"ROI bounding box: {roi_bbox.tolist()}")
    logger.info(f"Estimated encryption time: {est_total_time / 60:.1f} minutes (shots={shots})")
    logger.info(f"Analysis time: {elapsed:.2f}s")
    logger.info(f"Visualizations saved to: {analysis_dir}")
    logger.info("=" * 70)

    return {
        "image_info": image_info,
        "roi_pixels": int(np.sum(roi_mask)),
        "background_pixels": int(np.sum(background_mask)),
        "block_stats": block_stats,
        "roi_bbox": roi_bbox.tolist(),
        "estimated_encryption_minutes": est_total_time / 60,
        "analysis_time_seconds": elapsed,
        "visualization_files": saved_files,
    }
