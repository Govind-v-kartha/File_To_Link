"""
Verify Workflow - Standalone Verification.

Compares an original image with a decrypted image to verify zero data loss.
Can be run independently of the encryption/decryption pipeline.
"""

import os
import time
from datetime import datetime

from utils.logger import setup_logger, get_config_path, load_config
from utils.image_utils import load_image
from engines.verification_engine import verify_zero_data_loss, generate_verification_report

logger = setup_logger("VERIFY_WORKFLOW", get_config_path())


def run_verification(
    original_image_path: str,
    decrypted_image_path: str,
    output_dir: str = None,
    config: dict = None,
) -> dict:
    """
    Run standalone verification comparing original and decrypted images.

    Args:
        original_image_path: Path to the original image.
        decrypted_image_path: Path to the decrypted image.
        output_dir: Output directory for verification report.
        config: Configuration dict.

    Returns:
        Verification report dictionary.
    """
    start_time = time.time()

    if config is None:
        config = load_config()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(
            project_root, config["paths"]["output_dir"], "decrypted"
        )

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STANDALONE VERIFICATION MODE")
    logger.info("=" * 70)
    logger.info(f"Original:  {original_image_path}")
    logger.info(f"Decrypted: {decrypted_image_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Load images
    original = load_image(original_image_path)
    decrypted = load_image(decrypted_image_path)

    logger.info(f"Original shape:  {original.shape}")
    logger.info(f"Decrypted shape: {decrypted.shape}")

    # Run verification
    report = verify_zero_data_loss(original, decrypted)

    # Save report
    orig_basename = os.path.splitext(os.path.basename(original_image_path))[0]
    report_path = os.path.join(output_dir, f"verification_report_{orig_basename}.txt")
    generate_verification_report(report, report_path)

    elapsed = time.time() - start_time

    logger.info(f"\nVerification completed in {elapsed:.2f}s")
    logger.info(f"Report saved to: {report_path}")

    return report
