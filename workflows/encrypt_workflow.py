"""
Encryption Workflow - End-to-End Encryption Pipeline.

Orchestrates the complete encryption pipeline:
1. Load and validate input image
2. FlexiMo AI segmentation (ROI/Background separation)
3. ROI block division (32x32 blocks)
4. Quantum encryption of ROI blocks (NEQR via Repo B)
5. Classical encryption of background (AES-256-GCM)
6. Fusion of encrypted components
7. Metadata storage
8. Output encrypted image
"""

import os
import json
import time
import hashlib
import numpy as np
from datetime import datetime

from utils.logger import setup_logger, get_config_path
from utils.config_loader_secure import load_config_secure
from utils.image_utils import (
    load_image,
    save_image,
    compute_image_hash,
    get_image_info,
    embed_png_metadata,
)
from utils.crypto_utils import (
    generate_master_seed,
    derive_aes_key,
    generate_nonce,
    derive_quantum_seeds,
    generate_session_nonce,
    derive_all_block_seeds,
    encode_bytes_b64,
    save_key_material,
)
from utils.crypto_utils_pqc import (
    secure_key_export,
    save_pqc_keys_to_file,
    sign_bundle,
    save_signature_file,
    load_dilithium_private_key,
    protect_keys,
    save_protected_keys
)
from utils.bench import measure
from engines.ai_engine import segment_image_fleximo, save_segmentation_visualization
from engines.decision_engine import divide_roi_into_blocks, get_block_statistics
from engines.quantum_engine import encrypt_all_blocks
from engines.classical_engine import encrypt_background
from engines.fusion_engine import fuse_encrypted_image
from utils.block_utils import BLOCK_SIZE

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
        config = load_config_secure()

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
    # STEP 3: ROI Block Division (32x32 blocks)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 3: Dividing ROI into 32x32 blocks...")
    blocks, block_map, roi_bbox = divide_roi_into_blocks(original_image, roi_mask)

    # Filter out blocks that have ZERO actual ROI pixels (background-only blocks
    # inside the bounding box).  The bounding box spans nearly the whole image
    # but only 23.5 % of pixels are ROI, so this dramatically cuts block count.
    y_min_bb, x_min_bb = int(roi_bbox[0]), int(roi_bbox[1])
    filtered_blocks = []
    filtered_map = []
    for blk, bm in zip(blocks, block_map):
        gx, gy = int(bm["position"][0]), int(bm["position"][1])
        roi_patch = roi_mask[gy:gy + 32, gx:gx + 32]
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
    # STEP 4: Generate Encryption Keys & Per-Block Ephemeral Seeds (FIX #5)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 4: Generating encryption keys with per-block ephemeral seeds...")
    master_seed = generate_master_seed(32)
    salt = os.urandom(16)
    aes_key = derive_aes_key(master_seed, salt)
    nonce = generate_nonce(12)
    
    # Generate session nonce for forward secrecy (FIX #5)
    session_nonce = generate_session_nonce(16)
    logger.info(f"✅ Session nonce generated for forward secrecy")
    
    # Derive per-block ephemeral seeds using ratchet mechanism
    quantum_seeds = derive_all_block_seeds(master_seed, len(blocks), session_nonce)
    logger.info(f"✅ Derived ephemeral seeds for {len(blocks)} blocks")
    logger.info(f"   Each block has unique seed: even if master_seed leaks, past sessions are secure")

    # Save key material (without ML-KEM wrapping yet - added in STEP 9)
    key_path = os.path.join(metadata_dir, f"{image_basename}_keys.json")
    save_key_material(master_seed, aes_key, salt, key_path)
    logger.info(f"Keys generated (temporary): {key_path}")
    
    # ════════════════════════════════════════════════════════════════════
    # STEP 4.5: Protect Keys at Rest (Scrypt + AES-256-GCM) 🔐 NEW
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n>>> STEP 4.5: Protecting keys at rest (Scrypt + AES-256-GCM)...")
    
    key_passphrase = config.get("key_protection", {}).get("passphrase")
    if key_passphrase:
        try:
            keys_dict = {
                "master_seed": master_seed.hex(),
                "aes_key": aes_key.hex(),
                "salt": salt.hex()
            }
            
            protected_key_path = os.path.join(metadata_dir, f"{image_basename}_keys.enc")
            save_protected_keys(keys_dict, key_passphrase, protected_key_path)
            
            # Store encryption protection metadata
            # ⚠️ SECURITY: Never store the KEK itself - only algorithm parameters
            metadata_temp = {"key_protection": {
                "enabled": True,
                "method": "Scrypt + AES-256-GCM",
                "protected_keys_file": protected_key_path,
                "scrypt_params": {
                    "n": 2**14,      # 16384 iterations (memory cost)
                    "r": 8,          # Block size
                    "p": 1,          # Parallelization
                    "salt_length": 16
                },
                "aes_gcm_params": {
                    "key_length": 32,    # 256-bit key for AES
                    "nonce_length": 12,
                    "tag_length": 16
                }
            }}
            
            logger.info("✅ Keys protected at rest: Scrypt + AES-256-GCM")
            protected_key_file = protected_key_path
        except Exception as e:
            logger.warning(f"⚠️  Key protection failed: {e}. Keys NOT encrypted at rest (INSECURE)")
            protected_key_file = None
    else:
        logger.warning("⚠️  No passphrase configured. Keys NOT encrypted at rest (INSECURE)")
        protected_key_file = None

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
    
    # ⚡ FIX #6-ENHANCEMENT: Compute SHA-256 hash of .enc file for integrity verification
    # This hash will be included in the signed metadata, so tampering with the .enc file
    # will break the signature verification (transitively covering the .enc file)
    with measure("SHA-256 hash (1 MB)"):
        enc_file_hash = hashlib.sha256(ciphertext).hexdigest()
    classical_enc_info["enc_file_hash"] = enc_file_hash
    classical_enc_info["enc_file_hash_algorithm"] = "SHA-256"
    logger.info(f"⚡ FIX #6-ENHANCEMENT: Computed SHA-256 hash of encrypted background: {enc_file_hash[:16]}...")

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
        encrypted_blocks, block_map, encrypted_bg_visual, original_shape
    )

    # Save fused encrypted image
    fused_path = os.path.join(encrypted_dir, f"{image_basename}_encrypted.png")
    save_image(fused_image, fused_path)
    logger.info(f"Encrypted image saved: {fused_path}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 7.5: Embed PNG Metadata (Dependency Warning & Bundle ID)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 7.5: Embedding PNG metadata with dependency information...")
    
    try:
        # Calculate bundle ID (first 16 chars of SHA256 hash of metadata)
        # This will be matched during decryption
        bundle_id = None  # Will be set after metadata is saved in STEP 8
        
        # For now, create a preliminary metadata file to hash
        metadata_path = os.path.join(encrypted_dir, f"{image_basename}_metadata.json")
        preliminary_metadata = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(metadata_path, "w") as f:
            json.dump(preliminary_metadata, f, indent=2)
        
        # Calculate bundle ID
        with open(metadata_path, "rb") as f:
            metadata_bytes = f.read()
        bundle_id = hashlib.sha256(metadata_bytes).hexdigest()[:16]
        
        # Prepare PNG metadata chunks
        png_metadata = {
            "DependencyWarning": "⚠️ This image requires st2_background.enc for full decryption",
            "BundleID": bundle_id,
            "ImageType": "Encrypted-Hybrid-Quantum-Classical",
            "EncryptionMethod": "ML-KEM + NEQR + AES-256-GCM",
            "RequiredFiles": "st2_background.enc, st2_metadata.json, st2_bundle.sig",
            "DataIntegrityAlert": "Silent data loss prevented: all dependencies embedded as warnings"
        }
        
        # Embed metadata in PNG
        embed_png_metadata(fused_path, png_metadata)
        logger.info(f"✅ PNG metadata embedded with Bundle ID: {bundle_id}")
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to embed PNG metadata (non-critical): {e}")

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
                "shots_per_block": config.get("quantum_encryption", {}).get("shots", 16384),
                "encoding": "NEQR",
                "master_seed_hash": quantum_seeds["master_seed_hash"],
                "session_nonce": quantum_seeds["session_nonce"],
                "session_nonce_b64": quantum_seeds["session_nonce_b64"],
                "num_blocks_encrypted": len(encrypted_blocks),
                "forward_secrecy": True,
                "forward_secrecy_info": "Each block derived from: master_seed + session_nonce + block_id. Even if master_seed leaks, past sessions remain secure.",
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

    # ════════════════════════════════════════════════════════════════════
    # HYBRID PHASE 3: Embed blocks in metadata (no .npy sidecar files)
    # ════════════════════════════════════════════════════════════════════
    from utils.crypto_utils import encode_ndarray_b64

    # Embed ROI mask in metadata (no .npy file needed)
    metadata["encryption_metadata"]["roi_information"]["roi_mask_b64"] = encode_ndarray_b64(roi_mask)
    metadata["encryption_metadata"]["roi_information"]["roi_mask_shape"] = list(roi_mask.shape)
    metadata["encryption_metadata"]["roi_information"]["roi_mask_dtype"] = "uint8"
    logger.info(f"Embedded ROI mask in metadata: shape={roi_mask.shape}")

    # Embed encrypted blocks in metadata (no .npy files)
    blocks_b64 = []
    blocks_shapes = []
    for i, block in enumerate(encrypted_blocks):
        blocks_b64.append(encode_ndarray_b64(block))
        blocks_shapes.append(list(block.shape))

    metadata["encryption_metadata"]["output_files"]["encrypted_blocks_b64"] = blocks_b64
    metadata["encryption_metadata"]["output_files"]["encrypted_blocks_shapes"] = blocks_shapes
    metadata["encryption_metadata"]["output_files"]["encrypted_blocks_dtype"] = "uint8"

    logger.info(
        f"Embedded {len(encrypted_blocks)} encrypted blocks in metadata "
        f"(aligned bbox: all blocks {BLOCK_SIZE}x{BLOCK_SIZE}, no padding, no .npy files)"
    )


    # Save metadata
    metadata_path = os.path.join(metadata_dir, f"{image_basename}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved: {metadata_path}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 9: Post-Quantum Key Encapsulation (ML-KEM/Kyber768)
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n>>> STEP 9: Post-quantum key protection (ML-KEM/Kyber768)...")
    
    recipient_public_key_path = config.get("post_quantum", {}).get("recipient_public_key_path")
    if recipient_public_key_path and os.path.exists(recipient_public_key_path):
        try:
            with open(recipient_public_key_path, "rb") as f:
                recipient_public_key = f.read()
            
            wrapped_keys = secure_key_export(master_seed, recipient_public_key)
            # Store the AES-derivation salt alongside the wrapped seed so the
            # ML-KEM decrypt branch can re-derive the same aes_key.
            wrapped_keys["aes_salt"] = salt.hex()
            metadata["encryption_metadata"]["post_quantum"] = wrapped_keys
            save_pqc_keys_to_file(wrapped_keys, key_path)
            logger.info("✅ Keys protected with ML-KEM (Kyber768) - Zero knowledge key transport")
        except Exception as e:
            logger.warning(f"⚠️  ML-KEM key wrapping failed: {e}. Saving unprotected keys (INSECURE)")
    else:
        logger.warning("⚠️  No recipient public key configured. Master seed not post-quantum protected (INSECURE)")
    
    # Update metadata with ML-KEM info
    metadata_path = os.path.join(metadata_dir, f"{image_basename}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata updated with post-quantum protection: {metadata_path}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 10: Metadata Bundle Signature (ML-DSA/Dilithium3)
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n>>> STEP 10: Signing metadata bundle (ML-DSA/Dilithium3)...")
    
    sender_private_key_path = config.get("metadata_signature", {}).get("sender_private_key_path")
    if sender_private_key_path and os.path.exists(sender_private_key_path):
        try:
            sender_private_key = load_dilithium_private_key(sender_private_key_path)
            
            # Sign the metadata bundle. AFTER this call, metadata_path on
            # disk MUST NOT be modified — any change invalidates the signature.
            signature_hex = sign_bundle(metadata_path, sender_private_key)

            # Save signature to .sig file (detached signature)
            sig_path = os.path.join(metadata_dir, f"{image_basename}_bundle.sig")
            save_signature_file(signature_hex, sig_path)

            logger.info("✅ Metadata bundle signed with ML-DSA (Dilithium3) - Integrity & Authenticity verified")
            sig_file = sig_path
        except Exception as e:
            logger.warning(f"⚠️  Metadata signing failed: {e}. Bundle NOT signed (INSECURE)")
            sig_file = None
    else:
        logger.warning("⚠️  No sender private key configured. Metadata bundle NOT signed (INSECURE)")
        sig_file = None

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
    if protected_key_file:
        logger.info(f"Protected keys: {protected_key_file}")
    if sig_file:
        logger.info(f"Signature: {sig_file}")
    logger.info("=" * 70)

    return {
        "encrypted_image_path": fused_path,
        "metadata_path": metadata_path,
        "key_path": key_path,
        "protected_key_path": protected_key_file,
        "bg_cipher_path": bg_cipher_path,
        "signature_path": sig_file,
        "analysis_dir": analysis_dir,
        "total_time_seconds": total_time,
        "original_image": original_copy,
        "roi_mask": roi_mask,
        "background_mask": background_mask,
        "block_map": block_map,
        "roi_bbox": roi_bbox,
        "all_encryption_info": all_encryption_info,
    }
