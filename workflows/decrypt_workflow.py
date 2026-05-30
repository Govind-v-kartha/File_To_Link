"""
Decryption Workflow - End-to-End Decryption Pipeline.

Orchestrates the complete decryption pipeline to perfectly reconstruct
the original image from encrypted data:
1. Load encrypted data and metadata
2. Unfuse encrypted image into components
3. Classical decryption of background (AES-256-GCM)
4. Quantum decryption of ROI blocks (reverse NEQR)
5. Reconstruct full image from decrypted components
6. Verify zero data loss
"""

import os
import json
import time
import hashlib
import numpy as np
from datetime import datetime

from utils.logger import setup_logger, get_config_path
from utils.config_loader_secure import load_config_secure
from utils.image_utils import load_image, save_image, read_png_metadata, verify_png_dependencies
from utils.crypto_utils import load_key_material, derive_quantum_seeds, derive_all_block_seeds, decode_bytes_b64
from utils.crypto_utils_pqc import (
    secure_key_import,
    load_pqc_keys_from_file,
    verify_bundle,
    load_signature_file,
    load_dilithium_public_key,
    load_protected_keys
)
from utils.bench import measure
from utils.block_utils import reconstruct_from_blocks, place_roi_on_image
from engines.classical_engine import decrypt_background
from engines.quantum_engine import decrypt_all_blocks
from engines.fusion_engine import unfuse_encrypted_image
from engines.verification_engine import verify_zero_data_loss, generate_verification_report

logger = setup_logger("DECRYPT_WORKFLOW", get_config_path())


def run_decryption(
    metadata_path: str,
    output_dir: str = None,
    original_image_path: str = None,
    config: dict = None,
) -> dict:
    """
    Run the complete decryption pipeline.

    Args:
        metadata_path: Path to encryption metadata JSON file.
        output_dir: Output directory for decrypted files.
        original_image_path: Optional path to original image for verification.
        config: Configuration dict.

    Returns:
        Dictionary with decryption results and verification report.
    """
    total_start = time.time()

    if config is None:
        config = load_config_secure()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if output_dir is None:
        output_dir = os.path.join(project_root, config["paths"]["output_dir"])

    decrypted_dir = os.path.join(output_dir, "decrypted")
    os.makedirs(decrypted_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HYBRID AI-QUANTUM SATELLITE IMAGE ENCRYPTION SYSTEM")
    logger.info("DECRYPTION PIPELINE STARTED")
    logger.info("=" * 70)
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # ════════════════════════════════════════════════════════════════════
    # SIGNATURE VERIFICATION (SECURITY GATE - Before Any Decryption)
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n>>> SECURITY GATE: Verifying metadata bundle signature (ML-DSA/Dilithium3)...")
    
    # Construct expected .sig file path
    # Encryption saves as "{image_basename}_bundle.sig" while metadata is
    # "{image_basename}_metadata.json" — strip the _metadata suffix so the
    # signature filename matches what the encrypt step wrote.
    metadata_dir = os.path.dirname(metadata_path)
    metadata_basename = os.path.splitext(os.path.basename(metadata_path))[0]
    image_basename = metadata_basename[:-len("_metadata")] if metadata_basename.endswith("_metadata") else metadata_basename
    sig_path = os.path.join(metadata_dir, f"{image_basename}_bundle.sig")
    
    signature_verified = False
    if os.path.exists(sig_path):
        try:
            # Load sender's public key from config
            sender_public_key_path = config.get("metadata_signature", {}).get("sender_public_key_path")
            if sender_public_key_path and os.path.exists(sender_public_key_path):
                sender_public_key = load_dilithium_public_key(sender_public_key_path)
                signature_hex = load_signature_file(sig_path)
                
                # Verify bundle
                is_valid = verify_bundle(metadata_path, signature_hex, sender_public_key)
                
                if not is_valid:
                    raise RuntimeError("❌ SECURITY BREACH: Metadata signature verification FAILED - Bundle may be tampered!")
                
                signature_verified = True
                logger.info("✅ SECURITY GATE PASSED: Metadata bundle signature verified (ML-DSA Dilithium3)")
            else:
                logger.warning("⚠️  Sender public key not configured. Skipping signature verification (INSECURE)")
        except Exception as e:
            logger.error(f"❌ SECURITY GATE FAILED: {e}")
            raise RuntimeError(f"Metadata verification failed - Cannot proceed with decryption: {e}")
    else:
        logger.critical(f"🚨 SECURITY GATE FAILED: Signature file not found: {sig_path}")
        logger.critical("    A signed metadata bundle is required for decryption.")
        raise RuntimeError(
            f"Metadata signature file not found: {sig_path}. "
            f"Cannot proceed with decryption without an authenticated bundle."
        )

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Load Metadata and Encrypted Data
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 1: Loading encrypted data and metadata...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    enc_meta = metadata["encryption_metadata"]
    original_info = enc_meta["original_image"]
    original_shape = tuple([original_info["size"][1], original_info["size"][0], original_info["channels"]])
    # shape is (H, W, C) but size was stored as [W, H]

    block_map = enc_meta["block_map"]
    roi_bbox = np.array(enc_meta["roi_information"]["roi_bbox"])
    all_encryption_info = enc_meta["block_encryption_info"]

    # Load keys
    key_path = enc_meta["output_files"]["keys"]
    
    # ════════════════════════════════════════════════════════════════════
    # KEY RECOVERY BRANCH: Mutually exclusive paths
    # Three possible scenarios based on what's in metadata:
    #   1. post_quantum section exists → STEP 0: ML-KEM path (FIX #1)
    #   2. key_protection section exists → STEP 0.5: Scrypt + AES-GCM path (FIX #3)
    #   3. Neither exists → STEP 0-legacy: Plaintext keys (deprecated, v1.0 format)
    # ════════════════════════════════════════════════════════════════════
    
    pqc_keys = enc_meta.get("post_quantum")
    key_protection = enc_meta.get("key_protection")
    
    # ───────────────────────────────────────────────────────────────────
    # BRANCH 1: STEP 0 - Post-Quantum ML-KEM Key Recovery (FIX #1)
    # ───────────────────────────────────────────────────────────────────
    if pqc_keys:
        logger.info("\n>>> STEP 0: Post-Quantum ML-KEM Key Recovery (FIX #1)...")
        try:
            # Load recipient's ML-KEM private key from config
            config = load_config_secure()
            recipient_private_key_path = config.get("post_quantum", {}).get("recipient_private_key_path")
            
            if not recipient_private_key_path or not os.path.exists(recipient_private_key_path):
                raise FileNotFoundError(f"Recipient ML-KEM private key not found: {recipient_private_key_path}")
            
            with open(recipient_private_key_path, "rb") as f:
                recipient_private_key = f.read()
            
            # Recover master_seed from ML-KEM wrapped form
            master_seed = secure_key_import(
                kem_ciphertext=pqc_keys["kem_ciphertext"],
                wrapped_seed=pqc_keys["wrapped_seed"],
                wrap_nonce=pqc_keys["wrap_nonce"],
                recipient_private_key=recipient_private_key
            )
            logger.info("✅ Master seed recovered from ML-KEM (Kyber768) - Zero knowledge key transport")
            
            # Derive remaining keys from master_seed using the salt stored
            # alongside the ML-KEM wrapped seed during encryption.
            from utils.crypto_utils import derive_aes_key
            salt_hex = pqc_keys.get("aes_salt")
            if not salt_hex:
                raise RuntimeError("post_quantum.aes_salt missing from metadata — re-encrypt with current code")
            salt = bytes.fromhex(salt_hex)
            aes_key = derive_aes_key(master_seed, salt)
        except Exception as e:
            logger.error(f"❌ ML-KEM key recovery failed: {e}")
            raise RuntimeError(f"Post-quantum key recovery failed: {e}")
    
    # ───────────────────────────────────────────────────────────────────
    # BRANCH 2: STEP 0.5 - Protected Keys at Rest (FIX #3)
    # ───────────────────────────────────────────────────────────────────
    elif key_protection:
        logger.info("\n>>> STEP 0.5: Decrypt Protected Keys at Rest (FIX #3 - Scrypt + AES-256-GCM)...")
        try:
            protected_key_path = key_protection.get("protected_keys_file")
            if protected_key_path and os.path.exists(protected_key_path):
                # Get passphrase from config
                config = load_config_secure()
                key_passphrase = config.get("key_protection", {}).get("passphrase")
                if not key_passphrase:
                    raise ValueError("Key passphrase not configured. Cannot decrypt protected keys.")
                
                # Load and decrypt protected keys
                keys_dict = load_protected_keys(protected_key_path, key_passphrase)
                
                # Convert from hex strings back to bytes
                master_seed = bytes.fromhex(keys_dict.get("master_seed", ""))
                aes_key = bytes.fromhex(keys_dict.get("aes_key", ""))
                salt = bytes.fromhex(keys_dict.get("salt", ""))
                
                logger.info("✅ Keys decrypted from protected storage (Scrypt + AES-256-GCM)")
            else:
                logger.error("🚨 CRITICAL: Protected keys file not found. Cannot safely proceed.")
                raise FileNotFoundError(f"Protected keys file required but not found: {protected_key_path}")
        except Exception as e:
            logger.error(f"🚨 CRITICAL: Protected key decryption failed: {e}. Aborting decryption.")
            raise RuntimeError(f"Cannot decrypt without valid protected keys: {e}")
    
    # ───────────────────────────────────────────────────────────────────
    # BRANCH 3: STEP 0 (Fallback) - Plaintext Keys v1.0 Legacy (Deprecated)
    # ───────────────────────────────────────────────────────────────────
    else:
        logger.info("\n>>> STEP 0 (Fallback): Loading plaintext keys (v1.0 legacy format - DEPRECATED)...")
        logger.critical("🚨 CRITICAL SECURITY WARNING 🚨")
        logger.critical("    Raw key material found in PLAINTEXT on disk!")
        logger.critical("    This indicates an old encryption (v1.0 legacy format).")
        logger.critical("    Keys are NOT protected at rest and are VULNERABLE to theft.")
        logger.critical("    Proceeding with decryption, but STRONGLY recommend:")
        logger.critical("    1. Re-encrypt this image with modern protection (FIX #3: Scrypt + AES-256-GCM)")
        logger.critical("    2. Update to latest encryption version immediately")
        logger.critical("🚨 END WARNING 🚨\n")
        master_seed, aes_key, salt = load_key_material(key_path)

    # Load classical encryption info
    classical_info = enc_meta["classical_encryption"]
    nonce = decode_bytes_b64(classical_info["nonce"])
    tag = decode_bytes_b64(classical_info["tag"])
    bg_image_shape = tuple(classical_info["image_shape"])

    # ════════════════════════════════════════════════════════════════════
    # HYBRID PHASE 4: Load ROI mask from metadata (no .npy files)
    # ════════════════════════════════════════════════════════════════════
    from utils.crypto_utils import decode_ndarray_b64

    roi_mask_b64 = enc_meta["roi_information"].get("roi_mask_b64")
    if roi_mask_b64:
        roi_mask_shape = tuple(enc_meta["roi_information"]["roi_mask_shape"])
        roi_mask_dtype = enc_meta["roi_information"]["roi_mask_dtype"]
        roi_mask = decode_ndarray_b64(roi_mask_b64, roi_mask_shape, roi_mask_dtype)
        logger.info(f"Loaded ROI mask from metadata: {roi_mask_shape}")
    else:
        # Fallback for old format (backward compat)
        roi_mask_path = enc_meta["roi_information"].get("roi_mask_path")
        if roi_mask_path and os.path.exists(roi_mask_path):
            roi_mask = np.load(roi_mask_path)
            logger.info(f"Loaded ROI mask from {roi_mask_path} (old .npy format)")
        else:
            raise ValueError("ROI mask not found in metadata or file")

    # Load encrypted background
    bg_cipher_path = enc_meta["output_files"]["encrypted_background"]
    with open(bg_cipher_path, "rb") as f:
        ciphertext = f.read()

    # ⚡ FIX #6-ENHANCEMENT: Verify SHA-256 hash of encrypted background file
    # This ensures the signature (which covers metadata) transitively covers the .enc file
    classical_info = enc_meta.get("classical_encryption", {})
    expected_enc_file_hash = classical_info.get("enc_file_hash")
    
    if expected_enc_file_hash:
        with measure("SHA-256 hash (1 MB)"):
            computed_hash = hashlib.sha256(ciphertext).hexdigest()
        if computed_hash == expected_enc_file_hash:
            logger.info(f"✅ ⚡ FIX #6-ENHANCEMENT: Encrypted background file integrity verified (SHA-256 hash match)")
        else:
            logger.critical(f"🚨 CRITICAL: Encrypted background file integrity FAILED!")
            logger.critical(f"   Expected hash: {expected_enc_file_hash}")
            logger.critical(f"   Computed hash: {computed_hash}")
            logger.critical(f"   The .enc file may have been tampered with or corrupted!")
            raise RuntimeError(
                f"Background encryption file integrity check failed. "
                f"File may be corrupted or tampered. Aborting decryption."
            )
    else:
        logger.warning("⚠️  No SHA-256 hash in metadata. Unable to verify background file integrity.")
        logger.warning("    This may indicate an old encryption (v1.0 legacy) without FIX #6 protection.")

    # Load encrypted image
    encrypted_image_path = enc_meta["output_files"]["encrypted_image"]
    encrypted_image = load_image(encrypted_image_path)

    # ─────────────────────────────────────────────────────────────────
    # SECURITY CHECK: Verify PNG Dependency Metadata
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n🔍 Security Check: Verifying PNG dependencies...")
    try:
        png_verification = verify_png_dependencies(encrypted_image_path, metadata_path)
        
        if png_verification["has_metadata"]:
            logger.info("✅ PNG has embedded dependency metadata")
            if png_verification["has_dependency_warning"]:
                logger.info("⚠️  PNG dependency warning confirmed")
            if png_verification["has_bundle_id"]:
                logger.info(f"🔗 Bundle ID verified")
                if png_verification["bundle_id_matches"]:
                    logger.info("✅ Bundle ID matches metadata")
            if png_verification["required_files"]:
                logger.info(f"📦 Required files: {', '.join(png_verification['required_files'])}")
                # Verify that required files exist
                for req_file in png_verification["required_files"]:
                    req_path = os.path.join(os.path.dirname(encrypted_image_path), req_file)
                    if os.path.exists(req_path):
                        logger.info(f"  ✓ {req_file} found")
                    else:
                        logger.warning(f"  ⚠️  {req_file} NOT found (may cause data loss)")
        else:
            logger.warning("⚠️  PNG has no embedded dependency metadata - data loss risk")
            logger.warning("💡 Tip: Ensure st2_background.enc is preserved with this PNG")
    except Exception as e:
        logger.warning(f"⚠️  Could not verify PNG dependencies: {e}")

    logger.info(f"Metadata loaded: {len(block_map)} blocks, {len(all_encryption_info)} encryption records")
    logger.info(f"Original image shape: {original_shape}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Load Encrypted Blocks from Metadata (Pure Image-Only)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 2: Loading encrypted blocks from metadata...")

    # Load encrypted blocks from metadata (PRIMARY SOURCE - no .npy fallback)
    blocks_b64 = enc_meta["output_files"].get("encrypted_blocks_b64")
    if blocks_b64:
        blocks_shapes = enc_meta["output_files"]["encrypted_blocks_shapes"]
        blocks_dtype = enc_meta["output_files"]["encrypted_blocks_dtype"]
        
        encrypted_blocks = []
        for b64_str, shape in zip(blocks_b64, blocks_shapes):
            block = decode_ndarray_b64(b64_str, tuple(shape), blocks_dtype)
            encrypted_blocks.append(block)
        
        logger.info(
            f"Loaded {len(encrypted_blocks)} encrypted blocks from metadata "
            f"(aligned bbox: all {encrypted_blocks[0].shape[0]}×{encrypted_blocks[0].shape[1]}, no padding, image-only)"
        )
    else:
        # Fallback for old format (backward compat with .npy)
        enc_blocks_path = enc_meta["output_files"].get("encrypted_blocks")
        if enc_blocks_path and os.path.exists(enc_blocks_path):
            enc_blocks_array = np.load(enc_blocks_path)
            encrypted_blocks = [enc_blocks_array[i] for i in range(enc_blocks_array.shape[0])]
            logger.info(f"Loaded blocks from {enc_blocks_path} (old .npy format)")
        else:
            raise RuntimeError(
                "Encrypted blocks not found in metadata or .npy file. "
                "Cannot proceed with image-only decryption."
            )
    
    logger.info(f"Separated {len(encrypted_blocks)} encrypted blocks")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Classical Decryption of Background
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 3: Classical decryption of background (AES-256-GCM)...")
    decrypted_background = decrypt_background(
        ciphertext, tag, aes_key, nonce, bg_image_shape
    )
    logger.info(f"Background decrypted: {decrypted_background.shape}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Quantum Decryption of ROI Blocks (With Forward Secrecy - FIX #5)
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 4: Quantum decryption of ROI blocks with per-block ephemeral seeds...")
    
    # Check if forward secrecy is enabled (FIX #5)
    quantum_meta = enc_meta.get("quantum_encryption", {})
    forward_secrecy_enabled = quantum_meta.get("forward_secrecy", False)
    
    if forward_secrecy_enabled:
        # Derive per-block ephemeral seeds using session_nonce (FIX #5)
        session_nonce_b64 = quantum_meta.get("session_nonce_b64")
        if session_nonce_b64:
            session_nonce = decode_bytes_b64(session_nonce_b64)
            logger.info(f"🔐 Forward secrecy enabled: using session_nonce for block seeds")
            quantum_seeds = derive_all_block_seeds(master_seed, len(block_map), session_nonce)
            logger.info(f"✅ Per-block ephemeral seeds derived from session_nonce")
        else:
            logger.warning("⚠️  Forward secrecy enabled but session_nonce not found. Falling back to legacy derivation.")
            quantum_seeds = derive_quantum_seeds(master_seed, len(block_map))
    else:
        logger.info("ℹ️  Using legacy quantum seed derivation (no forward secrecy)")
        quantum_seeds = derive_quantum_seeds(master_seed, len(block_map))

    decrypted_blocks = decrypt_all_blocks(
        encrypted_blocks, quantum_seeds, all_encryption_info, config
    )
    logger.info(f"Quantum decryption complete: {len(decrypted_blocks)} blocks")

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: Reconstruct Full Image
    # ─────────────────────────────────────────────────────────────────
    logger.info("\n>>> STEP 5: Reconstructing full image...")

    # Start with decrypted background (full image with ROI pixels zeroed)
    reconstructed_image = decrypted_background.copy()

    # Use quantum-decrypted blocks directly (no bypass)
    reconstruction_blocks = decrypted_blocks

    # Reconstruct ROI from blocks
    roi_region = reconstruct_from_blocks(
        reconstruction_blocks, block_map, roi_bbox, original_shape
    )

    # Place ROI onto image — ONLY at ROI mask pixels, not the entire bbox.
    # The bbox may contain background pixels that were already correctly
    # decrypted in the background image; overwriting the full bbox would
    # destroy those pixels with zeros from the roi_region canvas.
    y_min, x_min, y_max, x_max = int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])

    if roi_region.ndim == 2:
        roi_region_3ch = np.stack([roi_region] * 3, axis=-1)
    else:
        roi_region_3ch = roi_region

    # Extract the ROI mask patch for the bounding box area
    roi_mask_patch = roi_mask[y_min:y_max, x_min:x_max]

    # Only overwrite pixels that are actually ROI
    roi_pixels = roi_mask_patch > 0
    if roi_pixels.ndim == 2 and roi_region_3ch.ndim == 3:
        roi_pixels_3d = np.stack([roi_pixels] * roi_region_3ch.shape[2], axis=-1)
    else:
        roi_pixels_3d = roi_pixels

    reconstructed_image[y_min:y_max, x_min:x_max][roi_pixels_3d] = roi_region_3ch[roi_pixels_3d]

    logger.info(f"Image reconstructed: {reconstructed_image.shape}")

    # Save masked reconstructed ROI (zero out non-ROI pixels for clean comparison)
    reconstructed_roi_masked = roi_region_3ch.copy()
    roi_mask_3d = np.stack([roi_mask_patch] * reconstructed_roi_masked.shape[2], axis=-1) if reconstructed_roi_masked.ndim == 3 else roi_mask_patch
    reconstructed_roi_masked[roi_mask_3d == 0] = 0
    roi_masked_path = os.path.join(decrypted_dir, "reconstructed_roi_masked.png")
    save_image(reconstructed_roi_masked, roi_masked_path)
    logger.info(f"Reconstructed ROI (masked) saved: {roi_masked_path}")

    # Save decrypted background
    bg_path = os.path.join(decrypted_dir, "decrypted_background.png")
    save_image(decrypted_background, bg_path)
    logger.info(f"Decrypted background saved: {bg_path}")

    # Save ROI mask as image
    roi_mask_path_out = os.path.join(decrypted_dir, "roi_mask.png")
    save_image((roi_mask * 255).astype(np.uint8), roi_mask_path_out)
    logger.info(f"ROI mask saved: {roi_mask_path_out}")

    # Force PNG output for lossless file storage (JPEG would introduce artifacts)
    original_filename = original_info.get("filename", "decrypted.png")
    basename_no_ext = os.path.splitext(original_filename)[0]
    decrypted_path = os.path.join(decrypted_dir, f"decrypted_{basename_no_ext}.png")
    save_image(reconstructed_image, decrypted_path)
    logger.info(f"Decrypted image saved (PNG, lossless): {decrypted_path}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: Verification (if original available)
    # ─────────────────────────────────────────────────────────────────
    verification_report = None
    if original_image_path and os.path.exists(original_image_path):
        logger.info("\n>>> STEP 6: Verifying zero data loss...")
        original_image = load_image(original_image_path)
        verification_report = verify_zero_data_loss(original_image, reconstructed_image)

        # Save verification report
        report_path = os.path.join(
            decrypted_dir, f"verification_report_{original_filename.split('.')[0]}.txt"
        )
        generate_verification_report(verification_report, report_path)
    else:
        logger.info("\n>>> STEP 6: Skipped verification (original image not provided)")

    # ─────────────────────────────────────────────────────────────────
    # COMPLETE
    # ─────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start

    logger.info("\n" + "=" * 70)
    logger.info("DECRYPTION PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.2f}s ({total_time / 60:.1f} minutes)")
    logger.info(f"Decrypted image: {decrypted_path}")
    if verification_report:
        logger.info(f"Verification: {verification_report['status']}")
    logger.info("=" * 70)

    return {
        "decrypted_image_path": decrypted_path,
        "reconstructed_image": reconstructed_image,
        "verification_report": verification_report,
        "total_time_seconds": total_time,
    }
