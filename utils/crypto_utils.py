"""
Cryptographic utilities for key generation and management.
Handles AES key derivation, seed management, and nonce generation.
"""

import os
import hashlib
import secrets
import base64
import json
from typing import Tuple

from utils.logger import setup_logger, get_config_path

logger = setup_logger("CRYPTO_UTILS", get_config_path())


def generate_master_seed(seed_length: int = 32) -> bytes:
    """
    Generate a cryptographically secure master seed.

    Args:
        seed_length: Length of the seed in bytes (default 32 = 256 bits).

    Returns:
        Random bytes of specified length.
    """
    seed = secrets.token_bytes(seed_length)
    logger.info(f"Generated master seed: {seed_length * 8} bits")
    return seed


def derive_aes_key(master_seed: bytes, salt: bytes = None) -> bytes:
    """
    Derive a 256-bit AES key from the master seed using PBKDF2.

    Args:
        master_seed: The master seed bytes.
        salt: Optional salt bytes. Generated randomly if not provided.

    Returns:
        32-byte AES-256 key.
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", master_seed, salt, 100000, dklen=32)
    logger.info("Derived AES-256 key from master seed")
    return key


def generate_nonce(nonce_size: int = 12) -> bytes:
    """
    Generate a random nonce for AES-GCM.

    Args:
        nonce_size: Size of nonce in bytes (default 12 = 96 bits for GCM).

    Returns:
        Random bytes of specified length.
    """
    nonce = secrets.token_bytes(nonce_size)
    logger.info(f"Generated random nonce: {nonce_size * 8} bits")
    return nonce


def derive_quantum_seeds(master_seed: bytes, num_blocks: int) -> dict:
    """
    Derive deterministic seeds for quantum encryption of each block.

    Uses HMAC-based key derivation so that the same master seed
    produces the same block seeds, ensuring deterministic encryption.

    Args:
        master_seed: The master seed bytes.
        num_blocks: Number of blocks to generate seeds for.

    Returns:
        Dictionary with quantum seed parameters.
    """
    # Derive x0, y0 for Henon map from master seed
    h = hashlib.sha512(master_seed + b"quantum_henon_seed").digest()
    x0 = (int.from_bytes(h[:8], "big") % 10000) / 10000.0
    y0 = (int.from_bytes(h[8:16], "big") % 10000) / 10000.0
    # Ensure valid Henon map initial conditions
    x0 = max(0.01, min(0.99, x0))
    y0 = max(0.01, min(0.99, y0))

    seeds = {
        "x0": x0,
        "y0": y0,
        "alpha": 1.4,
        "beta": 0.3,
        "num_blocks": num_blocks,
        "master_seed_hash": hashlib.sha256(master_seed).hexdigest(),
    }
    logger.info(
        f"Derived quantum seeds for {num_blocks} blocks: x0={x0:.6f}, y0={y0:.6f}"
    )
    return seeds


def encode_bytes_b64(data: bytes) -> str:
    """Encode bytes to base64 string for JSON storage."""
    return base64.b64encode(data).decode("utf-8")


def decode_bytes_b64(b64_str: str) -> bytes:
    """Decode base64 string back to bytes."""
    return base64.b64decode(b64_str.encode("utf-8"))


def save_key_material(
    master_seed: bytes,
    aes_key: bytes,
    salt: bytes,
    output_path: str,
) -> str:
    """
    Save key material to a secure JSON file.
    NOTE: In production, this should use a proper key management system.

    Args:
        master_seed: Master seed bytes.
        aes_key: AES key bytes.
        salt: Salt bytes used for key derivation.
        output_path: Path to save the key file.

    Returns:
        Path to the saved key file.
    """
    key_data = {
        "master_seed": encode_bytes_b64(master_seed),
        "aes_key": encode_bytes_b64(aes_key),
        "salt": encode_bytes_b64(salt),
        "key_size_bits": len(aes_key) * 8,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(key_data, f, indent=2)
    logger.info(f"Key material saved to {output_path}")
    return output_path


def load_key_material(key_path: str) -> Tuple[bytes, bytes, bytes]:
    """
    Load key material from a JSON file.

    Args:
        key_path: Path to the key file.

    Returns:
        Tuple of (master_seed, aes_key, salt).
    """
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Key file not found: {key_path}")
    with open(key_path, "r") as f:
        key_data = json.load(f)
    master_seed = decode_bytes_b64(key_data["master_seed"])
    aes_key = decode_bytes_b64(key_data["aes_key"])
    salt = decode_bytes_b64(key_data["salt"])
    logger.info(f"Key material loaded from {key_path}")
    return master_seed, aes_key, salt
