"""
Verification Engine - Zero Data Loss Verification.

Computes PSNR, SSIM, and pixel-perfect comparison metrics to verify
that encryption-decryption achieves zero data loss.
"""

import os
import numpy as np
from typing import Dict

from utils.logger import setup_logger, get_config_path
from utils.image_utils import compute_image_hash

logger = setup_logger("VERIFICATION_ENGINE", get_config_path())


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX^2 / MSE)
    If MSE = 0 (perfect match), returns infinity.

    Args:
        original: Original image array.
        reconstructed: Reconstructed image array.

    Returns:
        PSNR in dB (float('inf') for perfect match).
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_val = 255.0
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Uses the skimage implementation for accuracy.

    Args:
        original: Original image array.
        reconstructed: Reconstructed image array.

    Returns:
        SSIM value (1.0 = perfect match).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        if original.ndim == 3:
            return ssim(original, reconstructed, channel_axis=2, data_range=255)
        else:
            return ssim(original, reconstructed, data_range=255)
    except ImportError:
        logger.warning("skimage not available, computing simplified SSIM")
        # Simplified SSIM computation
        mu_x = np.mean(original.astype(np.float64))
        mu_y = np.mean(reconstructed.astype(np.float64))
        sigma_x = np.std(original.astype(np.float64))
        sigma_y = np.std(reconstructed.astype(np.float64))
        sigma_xy = np.mean(
            (original.astype(np.float64) - mu_x)
            * (reconstructed.astype(np.float64) - mu_y)
        )

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        ssim_val = (
            (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))

        return float(ssim_val)


def verify_zero_data_loss(
    original: np.ndarray, reconstructed: np.ndarray
) -> Dict:
    """
    Comprehensive verification that encryption-decryption achieves zero data loss.

    Checks:
    1. PSNR = infinity (MSE = 0)
    2. SSIM = 1.0
    3. Pixel-perfect match (max difference = 0)
    4. Hash comparison

    Args:
        original: Original image array (H, W, 3), uint8.
        reconstructed: Reconstructed image array (H, W, 3), uint8.

    Returns:
        Dictionary with all verification metrics and pass/fail status.
    """
    logger.info("=" * 60)
    logger.info("ZERO DATA LOSS VERIFICATION")
    logger.info("=" * 60)

    assert original.shape == reconstructed.shape, (
        f"Shape mismatch: original={original.shape}, reconstructed={reconstructed.shape}"
    )

    # Compute metrics
    psnr_val = compute_psnr(original, reconstructed)
    ssim_val = compute_ssim(original, reconstructed)

    # Pixel-perfect comparison
    diff = np.abs(original.astype(np.int16) - reconstructed.astype(np.int16))
    max_diff = int(np.max(diff))
    mean_diff = float(np.mean(diff))
    total_diff_pixels = int(np.sum(diff > 0))

    # Hash comparison
    hash_original = compute_image_hash(original)
    hash_reconstructed = compute_image_hash(reconstructed)
    hash_match = hash_original == hash_reconstructed

    # Determine pass/fail
    is_perfect = (
        psnr_val == float("inf")
        and ssim_val == 1.0
        and max_diff == 0
        and hash_match
    )

    # Build report
    report = {
        "status": "PASS" if is_perfect else "FAIL",
        "psnr_db": psnr_val if psnr_val != float("inf") else "Infinity",
        "ssim": ssim_val,
        "max_pixel_difference": max_diff,
        "mean_pixel_difference": mean_diff,
        "total_different_pixels": total_diff_pixels,
        "total_pixels": int(np.prod(original.shape)),
        "hash_original": hash_original,
        "hash_reconstructed": hash_reconstructed,
        "hash_match": hash_match,
        "is_pixel_perfect": max_diff == 0,
    }

    # Log results
    psnr_str = "∞ dB" if psnr_val == float("inf") else f"{psnr_val:.2f} dB"
    logger.info(f"PSNR  : {psnr_str}")
    logger.info(f"SSIM  : {ssim_val:.6f}")
    logger.info(f"Max pixel difference  : {max_diff}")
    logger.info(f"Mean pixel difference : {mean_diff:.6f}")
    logger.info(f"Different pixels      : {total_diff_pixels}/{np.prod(original.shape)}")
    logger.info(f"Hash match            : {hash_match}")
    logger.info(f"Hash original         : {hash_original[:16]}...")
    logger.info(f"Hash reconstructed    : {hash_reconstructed[:16]}...")

    if is_perfect:
        logger.info("✅ VERIFICATION PASSED: PSNR = ∞ dB, SSIM = 1.0000, Zero data loss confirmed")
    else:
        logger.error("❌ VERIFICATION FAILED: Data loss detected!")
        if psnr_val != float("inf"):
            logger.error(f"  PSNR is {psnr_str} (should be ∞)")
        if ssim_val != 1.0:
            logger.error(f"  SSIM is {ssim_val:.6f} (should be 1.0)")
        if max_diff > 0:
            logger.error(f"  Max pixel difference is {max_diff} (should be 0)")
        if not hash_match:
            logger.error("  Image hashes do not match")

    logger.info("=" * 60)

    return report


def generate_verification_report(
    report: Dict, output_path: str
) -> str:
    """
    Save the verification report as a text file.

    Args:
        report: Verification report dictionary.
        output_path: Path to save the report.

    Returns:
        Path to the saved report file.
    """
    import json

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "=" * 60,
        "HYBRID AI-QUANTUM SATELLITE IMAGE ENCRYPTION SYSTEM",
        "ZERO DATA LOSS VERIFICATION REPORT",
        "=" * 60,
        "",
        f"Status: {report['status']}",
        "",
        "--- Metrics ---",
        f"PSNR                  : {report['psnr_db']}",
        f"SSIM                  : {report['ssim']}",
        f"Max pixel difference  : {report['max_pixel_difference']}",
        f"Mean pixel difference : {report['mean_pixel_difference']:.6f}",
        f"Different pixels      : {report['total_different_pixels']}/{report['total_pixels']}",
        "",
        "--- Hash Verification ---",
        f"Hash original         : {report['hash_original']}",
        f"Hash reconstructed    : {report['hash_reconstructed']}",
        f"Hash match            : {report['hash_match']}",
        "",
        "--- Conclusion ---",
    ]

    if report["status"] == "PASS":
        lines.append("✅ ZERO DATA LOSS CONFIRMED: Perfect reconstruction achieved.")
        lines.append("   PSNR = ∞ dB, SSIM = 1.0000")
    else:
        lines.append("❌ DATA LOSS DETECTED: Reconstruction is not perfect.")

    lines.extend(["", "=" * 60])

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Also save as JSON
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Verification report saved to {output_path}")
    return output_path
