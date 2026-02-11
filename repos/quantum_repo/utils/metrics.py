# utils/metrics.py

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_psnr_ssim(original, reconstructed):
    """
    Computes PSNR and SSIM between original and reconstructed images.

    Parameters
    ----------
    original : np.ndarray (uint8)
        Original grayscale image (H x W)
    reconstructed : np.ndarray (uint8)
        Reconstructed grayscale image (H x W)

    Returns
    -------
    psnr_value : float
        Peak Signal-to-Noise Ratio (dB)
    ssim_value : float
        Structural Similarity Index
    """

    # Ensure correct type
    original = original.astype(np.uint8)
    reconstructed = reconstructed.astype(np.uint8)

    # Compute metrics
    psnr_value = psnr(original, reconstructed, data_range=255)
    ssim_value = ssim(original, reconstructed, data_range=255)

    return psnr_value, ssim_value
