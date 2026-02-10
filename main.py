"""
╔══════════════════════════════════════════════════════════════════╗
║   HYBRID AI-QUANTUM SATELLITE IMAGE ENCRYPTION SYSTEM          ║
║                                                                  ║
║   Master Orchestrator                                            ║
║                                                                  ║
║   Usage:                                                         ║
║     python main.py                                               ║
║       → Runs full pipeline: encrypt → decrypt → verify           ║
║         using the first image found in input/ (e.g. st1.png)     ║
║                                                                  ║
║   Optional modes (advanced):                                     ║
║     python main.py --mode encrypt   --input <image>              ║
║     python main.py --mode decrypt   --metadata <json>            ║
║     python main.py --mode analyze   --input <image>              ║
║     python main.py --mode verify    --original <img> --decrypted ║
║                                                                  ║
║   Repositories:                                                  ║
║     A) FlexiMo (AI Segmentation)                                 ║
║        https://github.com/danfenghong/IEEE_TGRS_Fleximo          ║
║     B) Quantum-image-encryption (NEQR)                           ║
║        https://github.com/ManavMNair/Quantum-image-encryption    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Set up project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Guard imports for multiprocessing spawn compatibility on Windows.
# Child processes re-import this module; the guard prevents them from
# running initialization code that depends on project-level modules.
if __name__ != "__mp_main__":
    from utils.logger import setup_logger, load_config
    logger = setup_logger("MAIN", os.path.join(PROJECT_ROOT, "config", "config.json"))


def verify_repositories() -> bool:
    """
    Verify that both external repositories are present and valid.

    Returns:
        True if both repos are valid.

    Raises:
        RuntimeError: If any repository is missing.
    """
    config = load_config()

    repos = {
        "FlexiMo (AI Segmentation)": config["repos"]["fleximo"]["path"],
        "Quantum Encryption (NEQR)": config["repos"]["quantum"]["path"],
    }

    all_valid = True
    for name, rel_path in repos.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.isdir(full_path):
            logger.info(f"✓ Repository found: {name} at {rel_path}")
        else:
            logger.error(f"✗ Repository NOT FOUND: {name} at {rel_path}")
            all_valid = False

    if not all_valid:
        raise RuntimeError(
            "Required repositories are missing! Please clone them:\n"
            "  git clone https://github.com/danfenghong/IEEE_TGRS_Fleximo repos/fleximo_repo\n"
            "  git clone https://github.com/ManavMNair/Quantum-image-encryption repos/quantum_repo"
        )

    return True


def verify_structure() -> bool:
    """Verify the project directory structure."""
    required_dirs = [
        "config",
        "engines",
        "workflows",
        "utils",
        "repos",
        "input",
        "output",
        "logs",
    ]

    for d in required_dirs:
        full_path = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(full_path):
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created missing directory: {d}")

    # Ensure output subdirectories
    for subdir in ["encrypted", "decrypted", "analysis", "metadata"]:
        os.makedirs(os.path.join(PROJECT_ROOT, "output", subdir), exist_ok=True)

    return True


def mode_encrypt(args):
    """Run encryption mode."""
    from workflows.encrypt_workflow import run_encryption

    config = load_config()

    # Determine input image(s)
    if args.input:
        image_path = args.input
        if not os.path.isabs(image_path):
            image_path = os.path.join(PROJECT_ROOT, image_path)
    else:
        # Look for images in input/ folder
        from utils.image_utils import list_input_images
        input_dir = os.path.join(PROJECT_ROOT, config["paths"]["input_dir"])
        images = list_input_images(input_dir)
        if not images:
            logger.error(
                f"No images found in {input_dir}/. "
                "Please place satellite images there or use --input <path>"
            )
            sys.exit(1)
        image_path = images[0]
        if len(images) > 1:
            logger.info(f"Multiple images found, processing first: {os.path.basename(image_path)}")

    logger.info(f"Encrypting: {image_path}")
    result = run_encryption(image_path, config=config, max_blocks=args.max_blocks)

    logger.info("\n" + "=" * 70)
    logger.info("ENCRYPTION SUMMARY")
    logger.info(f"  Encrypted image: {result['encrypted_image_path']}")
    logger.info(f"  Metadata: {result['metadata_path']}")
    logger.info(f"  Keys: {result['key_path']}")
    logger.info(f"  Time: {result['total_time_seconds']:.1f}s")
    logger.info("=" * 70)


def mode_decrypt(args):
    """Run decryption mode."""
    from workflows.decrypt_workflow import run_decryption

    config = load_config()

    # Find metadata file
    if args.metadata:
        metadata_path = args.metadata
        if not os.path.isabs(metadata_path):
            metadata_path = os.path.join(PROJECT_ROOT, metadata_path)
    else:
        # Look for metadata in output/metadata/
        metadata_dir = os.path.join(PROJECT_ROOT, config["paths"]["metadata_dir"])
        if os.path.isdir(metadata_dir):
            json_files = [
                f for f in os.listdir(metadata_dir)
                if f.endswith("_metadata.json")
            ]
            if json_files:
                metadata_path = os.path.join(metadata_dir, sorted(json_files)[0])
                logger.info(f"Using metadata: {metadata_path}")
            else:
                logger.error(f"No metadata files found in {metadata_dir}/")
                sys.exit(1)
        else:
            logger.error("No metadata directory found. Run encryption first.")
            sys.exit(1)

    # Determine original image for verification
    original_path = args.original if args.original else args.input

    result = run_decryption(
        metadata_path,
        original_image_path=original_path,
        config=config,
    )

    logger.info("\n" + "=" * 70)
    logger.info("DECRYPTION SUMMARY")
    logger.info(f"  Decrypted image: {result['decrypted_image_path']}")
    if result['verification_report']:
        report = result['verification_report']
        logger.info(f"  Verification: {report['status']}")
        psnr_str = "∞ dB" if report['psnr_db'] == "Infinity" else f"{report['psnr_db']}"
        logger.info(f"  PSNR: {psnr_str}")
        logger.info(f"  SSIM: {report['ssim']}")
    logger.info(f"  Time: {result['total_time_seconds']:.1f}s")
    logger.info("=" * 70)


def mode_analyze(args):
    """Run analysis mode."""
    from workflows.analyze_workflow import run_analysis

    config = load_config()

    if args.input:
        image_path = args.input
        if not os.path.isabs(image_path):
            image_path = os.path.join(PROJECT_ROOT, image_path)
    else:
        from utils.image_utils import list_input_images
        input_dir = os.path.join(PROJECT_ROOT, config["paths"]["input_dir"])
        images = list_input_images(input_dir)
        if not images:
            logger.error(f"No images found in {input_dir}/")
            sys.exit(1)
        image_path = images[0]

    result = run_analysis(image_path, config=config)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info(f"  ROI pixels: {result['roi_pixels']}")
    logger.info(f"  Background pixels: {result['background_pixels']}")
    logger.info(f"  Total blocks: {result['block_stats']['total_blocks']}")
    logger.info(f"  Estimated encryption time: {result['estimated_encryption_minutes']:.1f} minutes")
    logger.info("=" * 70)


def mode_verify(args):
    """Run verification mode."""
    from workflows.verify_workflow import run_verification

    if not args.original:
        logger.error("Verification requires --original <path_to_original_image>")
        sys.exit(1)

    if not args.decrypted:
        # Try to find in output/decrypted/
        config = load_config()
        decrypted_dir = os.path.join(PROJECT_ROOT, config["paths"]["output_dir"], "decrypted")
        if os.path.isdir(decrypted_dir):
            dec_files = [f for f in os.listdir(decrypted_dir) if f.startswith("decrypted_")]
            if dec_files:
                args.decrypted = os.path.join(decrypted_dir, sorted(dec_files)[0])
            else:
                logger.error(f"No decrypted images found in {decrypted_dir}/")
                sys.exit(1)
        else:
            logger.error("No decrypted images found. Run decryption first.")
            sys.exit(1)

    original_path = args.original
    decrypted_path = args.decrypted

    if not os.path.isabs(original_path):
        original_path = os.path.join(PROJECT_ROOT, original_path)
    if not os.path.isabs(decrypted_path):
        decrypted_path = os.path.join(PROJECT_ROOT, decrypted_path)

    report = run_verification(original_path, decrypted_path)

    status = report["status"]
    psnr_str = "∞ dB" if report["psnr_db"] == "Infinity" else f"{report['psnr_db']}"

    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info(f"  Status: {status}")
    logger.info(f"  PSNR: {psnr_str}")
    logger.info(f"  SSIM: {report['ssim']}")
    logger.info(f"  Max pixel diff: {report['max_pixel_difference']}")
    logger.info(f"  Hash match: {report['hash_match']}")
    logger.info("=" * 70)


def mode_full_pipeline(args):
    """
    Run the complete pipeline: encrypt → decrypt → verify.
    This is the default mode when no --mode is specified.
    """
    from workflows.encrypt_workflow import run_encryption
    from workflows.decrypt_workflow import run_decryption
    from workflows.verify_workflow import run_verification

    config = load_config()
    total_start = __import__("time").time()

    # ── Resolve input image ──────────────────────────────────────────
    if args.input:
        image_path = args.input
        if not os.path.isabs(image_path):
            image_path = os.path.join(PROJECT_ROOT, image_path)
    else:
        from utils.image_utils import list_input_images
        input_dir = os.path.join(PROJECT_ROOT, config["paths"]["input_dir"])
        images = list_input_images(input_dir)
        if not images:
            logger.error(
                f"No images found in {input_dir}/. "
                "Place a satellite image there or use --input <path>."
            )
            sys.exit(1)
        image_path = images[0]

    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    logger.info(f"Input image: {image_path}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1 — ENCRYPTION
    # ══════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  PHASE 1 / 3 — ENCRYPTION" + " " * 42 + "║")
    print("╚" + "═" * 68 + "╝")

    enc_result = run_encryption(image_path, config=config, max_blocks=args.max_blocks)

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 COMPLETE — ENCRYPTION")
    logger.info(f"  Encrypted image : {enc_result['encrypted_image_path']}")
    logger.info(f"  Metadata        : {enc_result['metadata_path']}")
    logger.info(f"  Keys            : {enc_result['key_path']}")
    logger.info(f"  Time            : {enc_result['total_time_seconds']:.1f}s")
    logger.info("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2 — DECRYPTION
    # ══════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  PHASE 2 / 3 — DECRYPTION" + " " * 42 + "║")
    print("╚" + "═" * 68 + "╝")

    dec_result = run_decryption(
        enc_result["metadata_path"],
        original_image_path=image_path,
        config=config,
    )

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 COMPLETE — DECRYPTION")
    logger.info(f"  Decrypted image : {dec_result['decrypted_image_path']}")
    logger.info(f"  Time            : {dec_result['total_time_seconds']:.1f}s")
    logger.info("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 3 — VERIFICATION
    # ══════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  PHASE 3 / 3 — VERIFICATION" + " " * 40 + "║")
    print("╚" + "═" * 68 + "╝")

    report = run_verification(image_path, dec_result["decrypted_image_path"])

    psnr_str = "∞ dB" if report["psnr_db"] == "Infinity" else f"{report['psnr_db']} dB"

    # ══════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    pipeline_time = __import__("time").time() - total_start

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  FINAL RESULTS" + " " * 53 + "║")
    print("╠" + "═" * 68 + "╣")

    logger.info("")
    logger.info("=" * 70)
    logger.info("  FULL PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Input image     : {image_path}")
    logger.info(f"  Encrypted image : {enc_result['encrypted_image_path']}")
    logger.info(f"  Decrypted image : {dec_result['decrypted_image_path']}")
    logger.info(f"  Metadata        : {enc_result['metadata_path']}")
    logger.info(f"  Keys            : {enc_result['key_path']}")
    logger.info("  ─────────────────────────────────────────────────────────")
    logger.info(f"  PSNR            : {psnr_str}")
    logger.info(f"  SSIM            : {report['ssim']:.6f}")
    logger.info(f"  Max pixel diff  : {report['max_pixel_difference']}")
    logger.info(f"  Hash match      : {report['hash_match']}")
    logger.info(f"  Status          : {report['status']}")
    logger.info("  ─────────────────────────────────────────────────────────")
    logger.info(f"  Encryption time : {enc_result['total_time_seconds']:.1f}s")
    logger.info(f"  Decryption time : {dec_result['total_time_seconds']:.1f}s")
    logger.info(f"  Total time      : {pipeline_time:.1f}s ({pipeline_time / 60:.1f} min)")
    logger.info("=" * 70)

    if report["status"] == "PASS":
        print("║  ✅  ZERO DATA LOSS CONFIRMED — PSNR = ∞, SSIM = 1.0       ║")
    else:
        print("║  ❌  DATA LOSS DETECTED — See verification report           ║")
    print("╚" + "═" * 68 + "╝")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid AI-Quantum Satellite Image Encryption System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python main.py                          Full pipeline (encrypt → decrypt → verify)
  python main.py --input input/st1.png    Full pipeline on a specific image

Advanced (single mode):
  python main.py --mode encrypt --input input/satellite.png
  python main.py --mode decrypt --metadata output/metadata/satellite_metadata.json
  python main.py --mode analyze --input input/satellite.png
  python main.py --mode verify  --original input/satellite.png --decrypted output/decrypted/decrypted_satellite.png
        """,
    )

    parser.add_argument(
        "--mode",
        required=False,
        default=None,
        choices=["encrypt", "decrypt", "analyze", "verify"],
        help="Operation mode (default: full pipeline = encrypt→decrypt→verify)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input satellite image (default: first image in input/)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to encryption metadata JSON (for decrypt mode)",
    )
    parser.add_argument(
        "--original",
        type=str,
        help="Path to original image (for verify/decrypt mode)",
    )
    parser.add_argument(
        "--decrypted",
        type=str,
        help="Path to decrypted image (for verify mode)",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Limit number of ROI blocks to encrypt (for quick testing)",
    )

    args = parser.parse_args()

    # Determine the effective mode
    effective_mode = args.mode if args.mode else "full"

    # Print banner
    mode_label = effective_mode.upper() if effective_mode != "full" else "FULL PIPELINE (ENCRYPT → DECRYPT → VERIFY)"
    print("=" * 70)
    print("  HYBRID AI-QUANTUM SATELLITE IMAGE ENCRYPTION SYSTEM")
    print(f"  Mode: {mode_label}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Startup checks
    logger.info("Running startup verification...")
    verify_structure()
    verify_repositories()

    # Dispatch to mode
    mode_handlers = {
        "full": mode_full_pipeline,
        "encrypt": mode_encrypt,
        "decrypt": mode_decrypt,
        "analyze": mode_analyze,
        "verify": mode_verify,
    }

    handler = mode_handlers[effective_mode]
    handler(args)


if __name__ == "__main__":
    main()
