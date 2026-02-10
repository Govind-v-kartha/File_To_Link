"""
Classical Engine - AES-256-GCM Encryption for Background Regions.

Encrypts the background (non-ROI) portion of satellite images using
AES-256-GCM (Galois/Counter Mode) for authenticated encryption.
"""

import os
import numpy as np
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from utils.logger import setup_logger, get_config_path
from utils.crypto_utils import encode_bytes_b64, decode_bytes_b64

logger = setup_logger("CLASSICAL_ENGINE", get_config_path())


def encrypt_background(
    image: np.ndarray,
    background_mask: np.ndarray,
    aes_key: bytes,
    nonce: bytes,
) -> Tuple[bytes, bytes, dict]:
    """
    Encrypt the background region of an image using AES-256-GCM.

    The background pixels (identified by background_mask) are extracted,
    serialized, and encrypted. ROI regions are zeroed out in the background
    image before encryption.

    Args:
        image: Original image (H, W, 3), dtype uint8.
        background_mask: Binary mask (H, W), 1 = background.
        aes_key: 32-byte (256-bit) AES key.
        nonce: 12-byte (96-bit) nonce for GCM.

    Returns:
        Tuple of:
            - ciphertext: Encrypted background bytes.
            - tag: 16-byte authentication tag (appended to ciphertext by AESGCM).
            - encryption_info: Dict with encryption details.
    """
    logger.info("=" * 60)
    logger.info("STARTING CLASSICAL ENCRYPTION OF BACKGROUND")
    logger.info("=" * 60)

    assert len(aes_key) == 32, f"AES key must be 32 bytes (256 bits), got {len(aes_key)}"
    assert len(nonce) == 12, f"Nonce must be 12 bytes (96 bits), got {len(nonce)}"

    # Extract background image (zero out ROI)
    background_image = image.copy()
    roi_mask = 1 - background_mask
    background_image[roi_mask > 0] = 0

    # Serialize to bytes
    bg_bytes = background_image.tobytes()
    logger.info(
        f"Background image serialized: {len(bg_bytes)} bytes, "
        f"shape={background_image.shape}"
    )

    # Encrypt using AES-256-GCM
    logger.info("Encrypting with AES-256-GCM...")
    aesgcm = AESGCM(aes_key)

    # AESGCM.encrypt returns ciphertext + tag (tag is appended, 16 bytes)
    ciphertext_with_tag = aesgcm.encrypt(nonce, bg_bytes, None)

    # Separate ciphertext and tag
    ciphertext = ciphertext_with_tag[:-16]
    tag = ciphertext_with_tag[-16:]

    encryption_info = {
        "algorithm": "AES-256-GCM",
        "key_size_bits": 256,
        "nonce": encode_bytes_b64(nonce),
        "tag": encode_bytes_b64(tag),
        "plaintext_size": len(bg_bytes),
        "ciphertext_size": len(ciphertext),
        "image_shape": list(background_image.shape),
    }

    logger.info(f"Background encrypted with AES-256-GCM")
    logger.info(f"  Plaintext size: {len(bg_bytes)} bytes")
    logger.info(f"  Ciphertext size: {len(ciphertext)} bytes")
    logger.info(f"  Authentication tag: {len(tag)} bytes")
    logger.info(f"  Nonce: {len(nonce)} bytes ({len(nonce) * 8} bits)")
    logger.info("=" * 60)

    return ciphertext, tag, encryption_info


def decrypt_background(
    ciphertext: bytes,
    tag: bytes,
    aes_key: bytes,
    nonce: bytes,
    image_shape: tuple,
) -> np.ndarray:
    """
    Decrypt the background region using AES-256-GCM.

    Args:
        ciphertext: Encrypted background bytes.
        tag: 16-byte authentication tag.
        aes_key: 32-byte AES key (same as used for encryption).
        nonce: 12-byte nonce (same as used for encryption).
        image_shape: Shape of the background image (H, W, 3).

    Returns:
        Decrypted background image (H, W, 3), dtype uint8.

    Raises:
        ValueError: If authentication fails (data tampered).
    """
    logger.info("=" * 60)
    logger.info("STARTING CLASSICAL DECRYPTION OF BACKGROUND")
    logger.info("=" * 60)

    assert len(aes_key) == 32, f"AES key must be 32 bytes, got {len(aes_key)}"
    assert len(nonce) == 12, f"Nonce must be 12 bytes, got {len(nonce)}"

    # Reconstruct ciphertext + tag for AESGCM
    ciphertext_with_tag = ciphertext + tag

    # Decrypt
    logger.info("Decrypting with AES-256-GCM...")
    aesgcm = AESGCM(aes_key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
    except Exception as e:
        raise ValueError(
            f"AES-GCM authentication failed: {e}. "
            "Data may have been tampered with or wrong key/nonce used."
        )

    # Reconstruct image
    background_image = np.frombuffer(plaintext, dtype=np.uint8).reshape(image_shape)

    logger.info(f"Background decrypted successfully: {background_image.shape}")
    logger.info(f"  Authentication tag verified: PASS")
    logger.info("=" * 60)

    return background_image
