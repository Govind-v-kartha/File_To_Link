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

# Enable PQC operation timing BEFORE importing any instrumented module.
# utils/bench.py reads BENCH_PQC at import time.
os.environ.setdefault("BENCH_PQC", "1")

# Set up project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Guard imports for multiprocessing spawn compatibility on Windows.
# Child processes re-import this module; the guard prevents them from
# running initialization code that depends on project-level modules.
if __name__ != "__mp_main__":
    from utils.logger import setup_logger
    from utils.config_loader_secure import load_config_secure
    from utils.security_manager import verify_all_permissions
    logger = setup_logger("MAIN", os.path.join(PROJECT_ROOT, "config", "config.json"))


def _emit_pqc_benchmark() -> None:
    """Flush PQC timings recorded during the run and print/save summary.

    Outputs three files at project root:
      benchmark_results.jsonl  raw per-call timings (appends across runs)
      benchmark_summary.txt    aggregate across all runs accumulated so far
      benchmark_results.tex    LaTeX rows ready to paste into ch5.tex Table 5.7
    """
    from utils.bench import flush, aggregate, emit_latex, TABLE_ORDER, is_enabled
    if not is_enabled():
        return

    jsonl = os.path.join(PROJECT_ROOT, "benchmark_results.jsonl")
    tex = os.path.join(PROJECT_ROOT, "benchmark_results.tex")
    summary_path = os.path.join(PROJECT_ROOT, "benchmark_summary.txt")

    flush(jsonl, run_label=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    if not os.path.exists(jsonl):
        return

    summary = aggregate(jsonl)
    if not summary:
        return
    emit_latex(summary, tex)

    # Determine number of distinct runs from the JSONL file
    n_runs = 0
    seen = set()
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["run"])
            except Exception:
                pass
    n_runs = len(seen)

    lines = []
    lines.append("=" * 70)
    lines.append(f"PQC OPERATION TIMINGS  ({n_runs} run(s) accumulated)")
    lines.append("=" * 70)
    lines.append(f"{'Operation':<28s} {'n':>4s} {'mean (ms)':>12s} {'std (ms)':>10s}")
    lines.append("-" * 70)
    total_mean = 0.0
    for op in TABLE_ORDER:
        if op not in summary:
            continue
        s = summary[op]
        total_mean += s["mean_ms"]
        lines.append(f"{op:<28s} {s['n']:>4d} {s['mean_ms']:>12.2f} {s['std_ms']:>10.2f}")
    lines.append("-" * 70)
    lines.append(f"{'Total PQC overhead':<28s} {'':>4s} {total_mean:>12.1f}")
    lines.append("")
    lines.append(f"  Raw timings → {jsonl}")
    lines.append(f"  LaTeX rows  → {tex}  (paste into ch5.tex Table 5.7)")
    if n_runs < 5:
        lines.append(f"  Note: std meaningful with n>=2; thesis table claims n=5.")
        lines.append(f"        Run main.py {5 - n_runs} more time(s) to reach n=5.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def verify_dependencies() -> bool:
    """
    Verify that all critical dependencies are available.
    
    Returns:
        True if all dependencies are available
        
    Raises:
        RuntimeError: If critical dependencies are missing
    """
    critical_missing = []
    optional_missing = []
    
    # Check critical imports
    critical_modules = {
        'torch': 'Deep Learning (PyTorch)',
        'cv2': 'Computer Vision (OpenCV)',
        'numpy': 'Numerical Computing',
        'PIL': 'Image Processing (Pillow)',
        'cryptography': 'Cryptography Library'
    }
    
    for module, name in critical_modules.items():
        try:
            __import__(module)
        except ImportError:
            critical_missing.append(f"{name} ({module})")
    
    # Check optional PQC import
    try:
        import oqs
    except BaseException as e:
        optional_missing.append(f"Post-Quantum Cryptography (liboqs-python): {str(e)[:60]}")
    
    # Report findings
    if critical_missing:
        logger.error("❌ CRITICAL DEPENDENCIES MISSING:")
        for dep in critical_missing:
            logger.error(f"   - {dep}")
        raise RuntimeError("Missing critical dependencies. Please run: pip install -r requirements.txt")
    
    if optional_missing:
        logger.warning("⚠️  OPTIONAL DEPENDENCIES MISSING:")
        for dep in optional_missing:
            logger.warning(f"   - {dep}")
        logger.warning("   Post-quantum encryption features will be UNAVAILABLE.")
        logger.warning("   Run: pip install --upgrade --force-reinstall liboqs-python")
    
    return True


def verify_repositories() -> bool:
    """
    Verify that both external repositories are present and valid.

    Returns:
        True if both repos are valid.

    Raises:
        RuntimeError: If any repository is missing.
    """
    config = load_config_secure()

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

    config = load_config_secure()

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
    _emit_pqc_benchmark()


def mode_decrypt(args):
    """Run decryption mode."""
    from workflows.decrypt_workflow import run_decryption

    config = load_config_secure()

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
    _emit_pqc_benchmark()


def mode_analyze(args):
    """Run analysis mode."""
    from workflows.analyze_workflow import run_analysis

    config = load_config_secure()

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
        config = load_config_secure()
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

    config = load_config_secure()
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

    _emit_pqc_benchmark()


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
    try:
        verify_dependencies()
    except RuntimeError as e:
        logger.error(f"Dependency check failed: {e}")
        sys.exit(1)
    
    # Load secure config with environment variable substitution
    config = load_config_secure()
    verify_structure()
    verify_repositories()
    
    # Verify file permissions for security-sensitive files
    verify_all_permissions(PROJECT_ROOT)

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
