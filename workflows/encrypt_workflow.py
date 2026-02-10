"""
Encryption Workflow - End-to-End Encryption Pipeline.

Orchestrates the complete encryption pipeline:
1. Load and validate input image
2. FlexiMo AI segmentation (ROI/Background separation)
3. ROI block division (8x8 blocks)
4. Quantum encryption of ROI blocks (NEQR via Repo B)
5. Classical encryption of background (AES-256-GCM)
6. Fusion of encrypted components
7. Metadata storage
8. Output encrypted image
"""

import os
import json
import time
import numpy as np
from datetime import datetime

from utils.logger import setup_logger, get_config_path, load_config
from utils.image_utils import (
    load_image,
    save_image,
    compute_image_hash,
    get_image_info,
)
from utils.crypto_utils import (
    generate_master_seed,
    derive_aes_key,
    generate_nonce,
    derive_quantum_seeds,
    encode_bytes_b64,
    save_key_material,
)
from engines.ai_engine import segment_image_fleximo, save_segmentation_visualization
from engines.decision_engine import divide_roi_into_blocks, get_block_statistics
from engines.quantum_engine import encrypt_all_blocks
from engines.classical_engine import encrypt_background
from engines.fusion_engine import fuse_encrypted_image

logger = setup_logger("ENCRYPT_WORKFLOW", get_config_path())


def run_encryption(
    image_path: str,
    output_dir: str = None,
    config: dict = None,
    max_blocks: int = None,
) -> dict:
    """
    Run the complete encryption pipeline on a single image.

    Args:
        image_path: Path to the input satellite image.
        output_dir: Output directory. If None, uses config default.
        config: Configuration dict. If None, loads from config.json.
        max_blocks: If set, limit the number of ROI blocks to encrypt.

    Returns:
        Dictionary with all output paths and metadata.
    """
    total_start = time.time()

    if config is None:
        config = load_config()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(project_root, config["paths"]["output_dir"])

    # Derived output directories
    encrypted_dir = os.path.join(output_dir, "encrypted")
    analysis_dir = os.path.join(output_dir, "analysis")
    metadata_dir = os.path.join(output_dir, "metadata")

    for d in [encrypted_dir, analysis_dir, metadata_dir]:
        os.makedirs(d, exist_ok=True)

    image_basename = os.path.splitext(os.path.basename(image_path))[0]

    logger.info("=" * 70)
    logger.info("HYBRID AI-QUANTUM SATELLITE IMAGE ENCRYPTION SYSTEM")
    logger.info("ENCRYPTION PIPELINE STARTED")
    logger.info("=" * 70)
    logger.info(f"Input image: {image_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Load and validate input image
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 1: Loading and validating input image...")
    original_image = load_image(image_path)
    original_shape = original_image.shape
    image_info = get_image_info(original_image, os.path.basename(image_path))
    logger.info(
        f"Image loaded: {original_shape[1]}x{original_shape[0]}, "
        f"{original_shape[2]} channels, hash={image_info['hash'][:16]}..."
    )

    # Save original copy for later verification
    original_copy = original_image.copy()

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: FlexiMo AI Semantic Segmentation
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 2: Running FlexiMo AI semantic segmentation...")
    roi_mask, background_mask, seg_raw = segment_image_fleximo(original_image, config)

    # Save segmentation visualizations
    save_segmentation_visualization(
        original_image, roi_mask, background_mask, analysis_dir, image_basename
    )
    logger.info(
        f"ROI: {np.sum(roi_mask)} pixels, Background: {np.sum(background_mask)} pixels"
    )

    # Free PyTorch/CUDA memory before quantum phase (ProcessPool needs RAM)
    import gc
    try:
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        del seg_raw  # raw segmentation output no longer needed
    except Exception:
        pass
    gc.collect()
    seg_raw = None  # keep variable alive but free the data

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: ROI Block Division (8x8 blocks)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 3: Dividing ROI into 8x8 blocks...")
    blocks, block_map, roi_bbox = divide_roi_into_blocks(original_image, roi_mask)

    # Filter out blocks that have ZERO actual ROI pixels (background-only blocks
    # inside the bounding box).  The bounding box spans nearly the whole image
    # but only 23.5 % of pixels are ROI, so this dramatically cuts block count.
    y_min_bb, x_min_bb = int(roi_bbox[0]), int(roi_bbox[1])
    filtered_blocks = []
    filtered_map = []
    for blk, bm in zip(blocks, block_map):
        gx, gy = int(bm["position"][0]), int(bm["position"][1])
        roi_patch = roi_mask[gy:gy + 8, gx:gx + 8]
        if roi_patch.any():                       # at least 1 ROI pixel
            filtered_blocks.append(blk)
            filtered_map.append(bm)
    skipped = len(blocks) - len(filtered_blocks)
    if skipped > 0:
        logger.info(
            f"⚡ Skipped {skipped} empty background blocks "
            f"({len(blocks)} → {len(filtered_blocks)} blocks, "
            f"{100 * skipped / len(blocks):.1f}% reduction)"
        )
    blocks, block_map = filtered_blocks, filtered_map

    # Optionally limit blocks for quick testing
    if max_blocks is not None and max_blocks < len(blocks):
        logger.info(f"⚡ --max-blocks={max_blocks}: truncating from {len(blocks)} to {max_blocks} blocks")
        blocks = blocks[:max_blocks]
        block_map = block_map[:max_blocks]

    block_stats = get_block_statistics(block_map)
    logger.info(
        f"Created {block_stats['total_blocks']} blocks "
        f"({block_stats['padded_blocks']} padded)"
    )

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Generate Encryption Keys
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 4: Generating encryption keys...")
    master_seed = generate_master_seed(32)
    salt = os.urandom(16)
    aes_key = derive_aes_key(master_seed, salt)
    nonce = generate_nonce(12)
    quantum_seeds = derive_quantum_seeds(master_seed, len(blocks))

    # Save key material
    key_path = os.path.join(metadata_dir, f"{image_basename}_keys.json")
    save_key_material(master_seed, aes_key, salt, key_path)
    logger.info(f"Keys generated and saved to {key_path}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: Quantum Encryption of ROI Blocks
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 5: Quantum encryption of ROI blocks (NEQR)...")
    logger.info(f"This will take a while - {len(blocks)} blocks to encrypt...")
    encrypted_blocks, all_encryption_info = encrypt_all_blocks(
        blocks, quantum_seeds, config
    )
    logger.info(f"Quantum encryption complete: {len(encrypted_blocks)} blocks encrypted")

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: Classical Encryption of Background
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 6: Classical encryption of background (AES-256-GCM)...")
    ciphertext, tag, classical_enc_info = encrypt_background(
        original_image, background_mask, aes_key, nonce
    )

    # Save encrypted background bytes
    bg_cipher_path = os.path.join(encrypted_dir, f"{image_basename}_background.enc")
    with open(bg_cipher_path, "wb") as f:
        f.write(ciphertext)
    logger.info(f"Encrypted background saved: {bg_cipher_path}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 7: Fusion of Encrypted Components
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 7: Fusing encrypted components...")

    # Create encrypted background image for fusion visualization
    # (The actual encrypted bytes are stored separately; for the fused image,
    # we create a visual representation)
    encrypted_bg_visual = np.frombuffer(
        ciphertext[: np.prod(original_shape)],
        dtype=np.uint8
    ).reshape(original_shape) if len(ciphertext) >= np.prod(original_shape) else np.random.randint(
        0, 256, original_shape, dtype=np.uint8
    )

    fused_image = fuse_encrypted_image(
        encrypted_blocks, block_map, encrypted_bg_visual, roi_mask, original_shape
    )

    # Save fused encrypted image
    fused_path = os.path.join(encrypted_dir, f"{image_basename}_encrypted.png")
    save_image(fused_image, fused_path)
    logger.info(f"Encrypted image saved: {fused_path}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 8: Save Metadata
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 8: Saving encryption metadata...")

    metadata = {
        "encryption_metadata": {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "original_image": image_info,
            "roi_information": {
                "total_roi_pixels": int(np.sum(roi_mask)),
                "total_background_pixels": int(np.sum(background_mask)),
                "total_blocks": block_stats["total_blocks"],
                "padded_blocks": block_stats["padded_blocks"],
                "roi_bbox": roi_bbox.tolist(),
            },
            "block_map": block_map,
            "quantum_encryption": {
                "backend": "AerSimulator",
                "shots_per_block": config.get("quantum_encryption", {}).get("shots", 1024),
                "encoding": "NEQR",
                "master_seed_hash": quantum_seeds["master_seed_hash"],
                "x0": quantum_seeds["x0"],
                "y0": quantum_seeds["y0"],
                "num_blocks_encrypted": len(encrypted_blocks),
            },
            "block_encryption_info": all_encryption_info,
            "classical_encryption": classical_enc_info,
            "output_files": {
                "encrypted_image": fused_path,
                "encrypted_background": bg_cipher_path,
                "keys": key_path,
            },
        }
    }

    # Save ROI mask
    roi_mask_path = os.path.join(metadata_dir, f"{image_basename}_roi_mask.npy")
    np.save(roi_mask_path, roi_mask)
    metadata["encryption_metadata"]["roi_information"]["roi_mask_path"] = roi_mask_path

    # Save background mask
    bg_mask_path = os.path.join(metadata_dir, f"{image_basename}_bg_mask.npy")
    np.save(bg_mask_path, background_mask)

    # Save encrypted blocks as numpy array for lossless decryption
    enc_blocks_path = os.path.join(metadata_dir, f"{image_basename}_encrypted_blocks.npy")
    enc_blocks_array = np.array(encrypted_blocks, dtype=np.uint8)
    np.save(enc_blocks_path, enc_blocks_array)
    metadata["encryption_metadata"]["output_files"]["encrypted_blocks"] = enc_blocks_path

    # Save original ROI blocks for verification
    original_blocks_path = os.path.join(metadata_dir, f"{image_basename}_original_blocks.npy")
    original_blocks_array = np.array(blocks, dtype=np.uint8)
    np.save(original_blocks_path, original_blocks_array)
    metadata["encryption_metadata"]["output_files"]["original_blocks"] = original_blocks_path

    # Save metadata
    metadata_path = os.path.join(metadata_dir, f"{image_basename}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved: {metadata_path}")

    # ─────────────────────────────────────────────────────────────────
    # COMPLETE
    # ─────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start

    logger.info("\n" + "=" * 70)
    logger.info("ENCRYPTION PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.2f}s ({total_time / 60:.1f} minutes)")
    logger.info(f"Encrypted image: {fused_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Keys: {key_path}")
    logger.info("=" * 70)

    return {
        "encrypted_image_path": fused_path,
        "metadata_path": metadata_path,
        "key_path": key_path,
        "bg_cipher_path": bg_cipher_path,
        "analysis_dir": analysis_dir,
        "total_time_seconds": total_time,
        "original_image": original_copy,
        "roi_mask": roi_mask,
        "background_mask": background_mask,
        "block_map": block_map,
        "roi_bbox": roi_bbox,
        "all_encryption_info": all_encryption_info,
    }
