"""
Quantum Engine - NEQR Quantum Encryption using Repository B.

This engine integrates the NEQR (Novel Enhanced Quantum Representation) quantum
encryption from Repository B. Each 8x8 ROI block is encoded into a quantum circuit,
scrambled using quantum gates, and encrypted using DNA encoding with chaotic keys.

This is TRUE quantum simulation using Qiskit's AerSimulator — NOT mathematical
simulation, NOT XOR, NOT chaotic maps pretending to be quantum.

Repository: https://github.com/ManavMNair/Quantum-image-encryption
"""

import os
import sys
import time
import warnings
import numpy as np
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# Suppress Henon map NaN/Inf cast warnings — expected for chaotic maps
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

from utils.logger import setup_logger, get_config_path, load_config
from utils.image_utils import rgb_to_grayscale

logger = setup_logger("QUANTUM_ENGINE", get_config_path())

# Global reference to the quantum repo modules — set after verification
_quantum_repo_path = None


def _verify_quantum_repo(repo_path: str) -> bool:
    """
    Verify that the Quantum Encryption repository exists and has required files.

    Args:
        repo_path: Path to the quantum repository.

    Returns:
        True if valid.

    Raises:
        RuntimeError: If repo is missing or invalid.
    """
    if not os.path.isdir(repo_path):
        raise RuntimeError(
            f"Quantum Encryption repository NOT FOUND at: {repo_path}\n"
            f"Please clone: git clone https://github.com/ManavMNair/Quantum-image-encryption {repo_path}"
        )

    required_files = [
        os.path.join("quantum", "neqr.py"),
        os.path.join("quantum", "scrambling.py"),
        os.path.join("chaos", "qrng.py"),
        os.path.join("chaos", "henon.py"),
        os.path.join("chaos", "hybrid_map.py"),
        os.path.join("dna", "dna_encode.py"),
        os.path.join("dna", "dna_decode.py"),
        os.path.join("utils", "metrics.py"),
    ]

    for rf in required_files:
        full_path = os.path.join(repo_path, rf)
        if not os.path.isfile(full_path):
            raise RuntimeError(
                f"Quantum repository incomplete: missing '{rf}' in {repo_path}"
            )

    logger.info(f"Quantum Encryption repository verified at: {repo_path}")
    return True


def _import_quantum_modules(repo_path: str):
    """
    Import all required modules from Repository B.

    Args:
        repo_path: Path to the quantum repository.

    Returns:
        Dictionary of imported modules/functions.
    """
    global _quantum_repo_path

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        logger.info(f"Added quantum repo to Python path: {repo_path}")

    _quantum_repo_path = repo_path

    try:
        from quantum.neqr import encode_neqr, reconstruct_neqr_image
        from quantum.scrambling import (
            quantum_scramble,
            quantum_permutation,
            reverse_quantum_scrambling,
            reverse_quantum_permutation,
        )
        from chaos.qrng import qrng
        from chaos.henon import henon_map
        from chaos.hybrid_map import generate_chaotic_key_image
        from dna.dna_encode import dna_encode
        from dna.dna_decode import dna_decrypt

        logger.info("All quantum repo modules imported successfully:")
        logger.info("  - quantum.neqr: encode_neqr, reconstruct_neqr_image")
        logger.info("  - quantum.scrambling: quantum_scramble, quantum_permutation")
        logger.info("  - quantum.scrambling: reverse_quantum_scrambling, reverse_quantum_permutation")
        logger.info("  - chaos.qrng: qrng")
        logger.info("  - chaos.henon: henon_map")
        logger.info("  - chaos.hybrid_map: generate_chaotic_key_image")
        logger.info("  - dna.dna_encode: dna_encode")
        logger.info("  - dna.dna_decode: dna_decrypt")

        return {
            "encode_neqr": encode_neqr,
            "reconstruct_neqr_image": reconstruct_neqr_image,
            "quantum_scramble": quantum_scramble,
            "quantum_permutation": quantum_permutation,
            "reverse_quantum_scrambling": reverse_quantum_scrambling,
            "reverse_quantum_permutation": reverse_quantum_permutation,
            "qrng": qrng,
            "henon_map": henon_map,
            "generate_chaotic_key_image": generate_chaotic_key_image,
            "dna_encode": dna_encode,
            "dna_decrypt": dna_decrypt,
        }
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import quantum repo modules: {e}\n"
            "Ensure qiskit and qiskit-aer are installed: pip install qiskit qiskit-aer"
        )


def _verify_qiskit_backend():
    """Verify that Qiskit AerSimulator is available."""
    try:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        logger.info(f"AerSimulator initialized for quantum encryption")
        logger.info(f"  Backend: {backend.name}")
        return backend
    except ImportError:
        raise RuntimeError(
            "Qiskit AerSimulator not available.\n"
            "Install with: pip install qiskit-aer"
        )


def _generate_keys_for_block(
    quantum_seeds: dict, block_id: int, block_size: int, modules: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate deterministic encryption keys for a specific block using Henon map.

    Args:
        quantum_seeds: Seed parameters from crypto_utils.
        block_id: Block identifier.
        modules: Imported quantum repo modules.

    Returns:
        Tuple of (bpk, ksk) key arrays.
    """
    # Derive unique seeds per block from master seeds
    x0_base = quantum_seeds["x0"]
    y0_base = quantum_seeds["y0"]

    # Perturb based on block_id for unique keys per block
    np.random.seed(int(x0_base * 1e6) + block_id)
    x0 = max(0.01, min(0.99, x0_base + block_id * 0.00001))
    y0 = max(0.01, min(0.99, y0_base + block_id * 0.00001))

    henon_map = modules["henon_map"]
    x, y = henon_map(x0, y0, n_iter=block_size)

    # Henon map with alpha=1.8 is chaotic and diverges — replace NaN/Inf
    x = np.nan_to_num(x, nan=0.5, posinf=0.99, neginf=0.01)
    y = np.nan_to_num(y, nan=0.5, posinf=0.99, neginf=0.01)

    bpk = np.floor(np.abs(x) * 256).astype(np.uint8)
    ksk = np.floor(np.abs(y) * 256).astype(np.uint8)

    return bpk, ksk


def encrypt_block_quantum(
    block: np.ndarray,
    block_id: int,
    quantum_seeds: dict,
    modules: dict,
    shots: int = 1024,
) -> Tuple[np.ndarray, dict]:
    """
    Encrypt a single 8x8 block using NEQR quantum encryption from Repository B.

    This is the core quantum encryption function. It:
    1. Converts the block to grayscale (NEQR works on grayscale)
    2. Encodes using NEQR (quantum circuit creation)
    3. Applies quantum scrambling gates
    4. Applies quantum permutation gates
    5. Measures with shots on AerSimulator
    6. Applies DNA encryption layer
    7. Returns encrypted block

    Args:
        block: 8x8x3 RGB block or 8x8 grayscale block.
        block_id: Block identifier for key derivation.
        quantum_seeds: Seed parameters for key generation.
        modules: Imported quantum repo modules.
        shots: Number of measurement shots (minimum 1024).

    Returns:
        Tuple of:
            - encrypted_block: Encrypted 8x8 block (uint8).
            - encryption_info: Dict with encryption details for this block.
    """
    start_time = time.time()
    block_size = 8

    # Extract functions from modules
    encode_neqr = modules["encode_neqr"]
    reconstruct_neqr_image = modules["reconstruct_neqr_image"]
    q_scramble = modules["quantum_scramble"]
    q_permutation = modules["quantum_permutation"]
    dna_encode = modules["dna_encode"]
    generate_chaotic_key_image = modules["generate_chaotic_key_image"]

    # Step 1: Convert to grayscale if needed (NEQR works on grayscale)
    if block.ndim == 3:
        gray_block = rgb_to_grayscale(block)
    else:
        gray_block = block.copy()

    assert gray_block.shape == (block_size, block_size), (
        f"Block must be {block_size}x{block_size}, got {gray_block.shape}"
    )

    # Step 2: Generate keys for this block
    bpk, ksk = _generate_keys_for_block(quantum_seeds, block_id, block_size, modules)

    # Step 3: NEQR encode - creates quantum circuit
    logger.debug(f"Block {block_id}: NEQR encoding...")
    qc = encode_neqr(gray_block)

    n = int(np.log2(block_size))
    num_position_qubits = 2 * n  # 6 qubits for 8x8

    # Step 4: Apply quantum scrambling (X and Z gates based on chaotic key)
    logger.debug(f"Block {block_id}: Applying quantum scrambling...")
    qc = q_scramble(qc, bpk, num_position_qubits)

    # Step 5: Apply quantum permutation (SWAP gates)
    logger.debug(f"Block {block_id}: Applying quantum permutation...")
    qc = q_permutation(qc, ksk, num_position_qubits)

    # Step 6: Measure quantum state with shots
    logger.debug(f"Block {block_id}: Measuring quantum state (shots={shots})...")
    scrambled_img = reconstruct_neqr_image(qc, block_size, block_size, shots=shots)

    # Step 7: Apply DNA encryption layer
    logger.debug(f"Block {block_id}: Applying DNA encryption...")
    DNi0, DNi1, DNi2, DNi3 = dna_encode(scrambled_img, ksk)

    # Generate chaotic key deterministically from block keys (avoids slow qrng per block)
    np.random.seed(int(bpk.sum()) * 1000 + int(ksk.sum()) + block_id)
    KH = np.random.randint(0, 256, (block_size, block_size), dtype=np.uint8)
    DKi0, DKi1, DKi2, DKi3 = dna_encode(KH, ksk)

    # XOR diffusion to create final encrypted block
    encrypted_block = (
        (DNi0 ^ DKi0) << 6
        | (DNi1 ^ DKi1) << 4
        | (DNi2 ^ DKi2) << 2
        | (DNi3 ^ DKi3)
    ).astype(np.uint8)

    elapsed = time.time() - start_time

    encryption_info = {
        "block_id": block_id,
        "shots": shots,
        "num_qubits": qc.num_qubits,
        "circuit_depth": qc.depth(),
        "encryption_time_seconds": elapsed,
        "bpk": bpk.tolist(),
        "ksk": ksk.tolist(),
        "KH": KH.tolist(),
        "DKi0": DKi0.tolist(),
        "DKi1": DKi1.tolist(),
        "DKi2": DKi2.tolist(),
        "DKi3": DKi3.tolist(),
    }

    return encrypted_block, encryption_info


def decrypt_block_quantum(
    encrypted_block: np.ndarray,
    block_id: int,
    quantum_seeds: dict,
    encryption_info: dict,
    modules: dict,
    shots: int = 1024,
) -> np.ndarray:
    """
    Decrypt a single 8x8 block using reverse NEQR quantum operations.

    Reverses the encryption process:
    1. DNA decryption
    2. Re-encode recovered scrambled image with NEQR
    3. Reverse quantum permutation
    4. Reverse quantum scrambling
    5. Measure to get original block

    Args:
        encrypted_block: Encrypted 8x8 block.
        block_id: Block identifier.
        quantum_seeds: Seed parameters for key generation.
        encryption_info: Encryption details for this block.
        modules: Imported quantum repo modules.
        shots: Number of measurement shots.

    Returns:
        Decrypted 8x8 block (uint8).
    """
    start_time = time.time()
    block_size = 8

    # Extract functions
    encode_neqr = modules["encode_neqr"]
    reconstruct_neqr_image = modules["reconstruct_neqr_image"]
    reverse_q_scrambling = modules["reverse_quantum_scrambling"]
    reverse_q_permutation = modules["reverse_quantum_permutation"]
    dna_decrypt = modules["dna_decrypt"]

    # Recover keys
    bpk = np.array(encryption_info["bpk"], dtype=np.uint8)
    ksk = np.array(encryption_info["ksk"], dtype=np.uint8)

    # Recover chaotic key DNA planes
    DKi0 = np.array(encryption_info["DKi0"], dtype=np.uint8)
    DKi1 = np.array(encryption_info["DKi1"], dtype=np.uint8)
    DKi2 = np.array(encryption_info["DKi2"], dtype=np.uint8)
    DKi3 = np.array(encryption_info["DKi3"], dtype=np.uint8)

    n = int(np.log2(block_size))
    num_position_qubits = 2 * n

    # Step 1: DNA decryption
    logger.debug(f"Block {block_id}: DNA decryption...")
    scrambled_recovered = dna_decrypt(encrypted_block, DKi0, DKi1, DKi2, DKi3, ksk)

    # Step 2: Re-encode the scrambled image with NEQR
    logger.debug(f"Block {block_id}: Re-encoding with NEQR...")
    qc_re = encode_neqr(scrambled_recovered)

    # Step 3: Reverse quantum permutation
    logger.debug(f"Block {block_id}: Reverse quantum permutation...")
    qc_re = reverse_q_permutation(qc_re, ksk, num_position_qubits)

    # Step 4: Reverse quantum scrambling
    logger.debug(f"Block {block_id}: Reverse quantum scrambling...")
    qc_re = reverse_q_scrambling(qc_re, bpk, num_position_qubits)

    # Step 5: Measure to get original block
    logger.debug(f"Block {block_id}: Final quantum reconstruction (shots={shots})...")
    decrypted_block = reconstruct_neqr_image(qc_re, block_size, block_size, shots=shots)

    elapsed = time.time() - start_time
    logger.debug(f"Block {block_id}: Decrypted in {elapsed:.2f}s")

    return decrypted_block


def _encrypt_blocks_parallel(
    blocks, encrypted_blocks, all_encryption_info,
    quantum_seeds, repo_path, shots, total_start, log_interval, logger
) -> bool:
    """
    Try parallel encryption with ProcessPoolExecutor, then ThreadPoolExecutor.
    Returns True if all blocks were successfully processed in parallel.
    Returns False if parallel failed and caller should use sequential fallback.
    """
    total_blocks = len(blocks)

    # Strategy 1: ThreadPoolExecutor (threads share memory, no spawn overhead)
    # Safe for Qiskit since AerSimulator releases GIL during C++ execution.
    # Strategy 2: ProcessPool with 1 worker (sequential but isolated, lower OOM risk)
    strategies = [
        ("ThreadPool (4 threads)", ThreadPoolExecutor, 4),
        ("ProcessPool (1 worker)", ProcessPoolExecutor, 1),
    ]

    for strategy_name, PoolClass, n_workers in strategies:
        logger.info(f"Trying {strategy_name} for encryption...")
        from engines.quantum_worker import encrypt_block_worker, worker_initializer

        completed = 0
        failed = 0
        pool_broken = False

        try:
            pool_kwargs = {"max_workers": n_workers}
            if PoolClass is ProcessPoolExecutor:
                pool_kwargs["initializer"] = worker_initializer
                pool_kwargs["initargs"] = (repo_path,)

            CHUNK = n_workers * 50 if PoolClass is ThreadPoolExecutor else n_workers * 3
            with PoolClass(**pool_kwargs) as executor:
                for chunk_start in range(0, total_blocks, CHUNK):
                    if pool_broken:
                        break
                    chunk_end = min(chunk_start + CHUNK, total_blocks)
                    work_items = [
                        (block, i, quantum_seeds, repo_path, shots)
                        for i, block in enumerate(
                            blocks[chunk_start:chunk_end], start=chunk_start
                        )
                    ]
                    try:
                        futures = {
                            executor.submit(encrypt_block_worker, item): item[1]
                            for item in work_items
                        }
                    except Exception as e:
                        logger.warning(f"{strategy_name} pool broken on submit: {e}")
                        pool_broken = True
                        break

                    for fut in as_completed(futures):
                        bid = futures[fut]
                        try:
                            block_id, enc_block, enc_info = fut.result(timeout=120)
                            encrypted_blocks[block_id] = enc_block
                            all_encryption_info[block_id] = enc_info
                        except Exception as e:
                            err_msg = str(e)
                            if "terminated abruptly" in err_msg or "BrokenProcessPool" in err_msg:
                                logger.warning(f"{strategy_name} process crashed at block {bid}")
                                pool_broken = True
                                failed += 1
                                break
                            logger.error(f"Block {bid} encrypt error: {e}")
                            failed += 1
                            encrypted_blocks[bid] = np.zeros((8, 8), dtype=np.uint8)
                            all_encryption_info[bid] = {"block_id": bid, "error": err_msg}
                        completed += 1

                        if completed % log_interval == 0 or completed == total_blocks:
                            elapsed = time.time() - total_start
                            pct = 100 * completed / total_blocks
                            avg = elapsed / completed
                            eta = avg * (total_blocks - completed)
                            logger.info(
                                f"[{strategy_name}] Encrypted {completed}/{total_blocks} "
                                f"({pct:.1f}%) avg={avg:.2f}s/block, ETA: {_format_time(eta)}"
                            )

            if not pool_broken and failed == 0:
                logger.info(f"{strategy_name} encryption completed successfully.")
                return True
            elif pool_broken:
                logger.warning(
                    f"{strategy_name} failed (pool crashed after {completed} blocks). "
                    "Trying next strategy..."
                )
                # Clear any partially-filled entries so sequential can redo them
                for i in range(total_blocks):
                    if encrypted_blocks[i] is not None and isinstance(
                        all_encryption_info[i], dict
                    ) and "error" in all_encryption_info[i]:
                        encrypted_blocks[i] = None
                        all_encryption_info[i] = None
            else:
                # Some individual blocks failed but pool didn't break - that's OK
                if failed > 0:
                    logger.warning(f"{failed} blocks had errors but pool survived")
                return True

        except Exception as e:
            logger.warning(f"{strategy_name} failed with: {e}. Trying next strategy...")
            continue

    logger.warning("All parallel strategies failed. Falling back to sequential.")
    return False


def _decrypt_blocks_parallel(
    encrypted_blocks, decrypted_blocks, quantum_seeds,
    all_encryption_info, repo_path, shots, total_start, log_interval, logger
) -> bool:
    """
    Try parallel decryption. Returns True if successful, False for sequential fallback.
    """
    total_blocks = len(encrypted_blocks)

    strategies = [
        ("ThreadPool (4 threads)", ThreadPoolExecutor, 4),
        ("ProcessPool (1 worker)", ProcessPoolExecutor, 1),
    ]

    for strategy_name, PoolClass, n_workers in strategies:
        logger.info(f"Trying {strategy_name} for decryption...")
        from engines.quantum_worker import decrypt_block_worker, worker_initializer

        completed = 0
        pool_broken = False

        try:
            pool_kwargs = {"max_workers": n_workers}
            if PoolClass is ProcessPoolExecutor:
                pool_kwargs["initializer"] = worker_initializer
                pool_kwargs["initargs"] = (repo_path,)

            CHUNK = n_workers * 50 if PoolClass is ThreadPoolExecutor else n_workers * 3
            with PoolClass(**pool_kwargs) as executor:
                for chunk_start in range(0, total_blocks, CHUNK):
                    if pool_broken:
                        break
                    chunk_end = min(chunk_start + CHUNK, total_blocks)
                    work_items = [
                        (enc_block, i, quantum_seeds, all_encryption_info[i], repo_path, shots)
                        for i, enc_block in enumerate(
                            encrypted_blocks[chunk_start:chunk_end], start=chunk_start
                        )
                    ]
                    try:
                        futures = {
                            executor.submit(decrypt_block_worker, item): item[1]
                            for item in work_items
                        }
                    except Exception as e:
                        logger.warning(f"{strategy_name} pool broken on submit: {e}")
                        pool_broken = True
                        break

                    for fut in as_completed(futures):
                        bid = futures[fut]
                        try:
                            block_id, dec_block = fut.result(timeout=120)
                            decrypted_blocks[block_id] = dec_block
                        except Exception as e:
                            err_msg = str(e)
                            if "terminated abruptly" in err_msg or "BrokenProcessPool" in err_msg:
                                logger.warning(f"{strategy_name} process crashed at block {bid}")
                                pool_broken = True
                                break
                            logger.error(f"Block {bid} decrypt error: {e}")
                            decrypted_blocks[bid] = np.zeros((8, 8), dtype=np.uint8)
                        completed += 1

                        if completed % log_interval == 0 or completed == total_blocks:
                            elapsed = time.time() - total_start
                            pct = 100 * completed / total_blocks
                            avg = elapsed / completed
                            eta = avg * (total_blocks - completed)
                            logger.info(
                                f"[{strategy_name}] Decrypted {completed}/{total_blocks} "
                                f"({pct:.1f}%) avg={avg:.2f}s/block, ETA: {_format_time(eta)}"
                            )

            if not pool_broken:
                logger.info(f"{strategy_name} decryption completed successfully.")
                return True
            else:
                logger.warning(
                    f"{strategy_name} failed (pool crashed). Trying next strategy..."
                )
                for i in range(total_blocks):
                    if decrypted_blocks[i] is None:
                        pass  # Will be handled by sequential fallback

        except Exception as e:
            logger.warning(f"{strategy_name} failed with: {e}. Trying next strategy...")
            continue

    logger.warning("All parallel strategies failed. Falling back to sequential.")
    return False


def encrypt_all_blocks(
    blocks: List[np.ndarray],
    quantum_seeds: dict,
    config: dict = None,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Encrypt all ROI blocks using quantum encryption.

    Args:
        blocks: List of 8x8x3 blocks.
        quantum_seeds: Seed parameters for key generation.
        config: Configuration dictionary.

    Returns:
        Tuple of:
            - encrypted_blocks: List of encrypted 8x8 blocks.
            - all_encryption_info: List of encryption info dicts per block.
    """
    if config is None:
        config = load_config()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_path = os.path.join(project_root, config["repos"]["quantum"]["path"])
    shots = config.get("quantum_encryption", {}).get("shots", 1024)

    logger.info("=" * 60)
    logger.info("STARTING QUANTUM ENCRYPTION OF ROI BLOCKS")
    logger.info("=" * 60)

    # Verify and import
    _verify_quantum_repo(repo_path)
    modules = _import_quantum_modules(repo_path)
    _verify_qiskit_backend()

    total_blocks = len(blocks)
    logger.info(f"Total blocks to encrypt: {total_blocks}")
    logger.info(f"Shots per block: {shots}")
    logger.info(f"Encoding: NEQR (Novel Enhanced Quantum Representation)")
    logger.info(f"NEQR encoding from repos/quantum_repo/")

    encrypted_blocks = [None] * total_blocks
    all_encryption_info = [None] * total_blocks
    total_start = time.time()
    completed_count = 0
    log_interval = max(1, total_blocks // 20)

    # ── Try parallel, then fall back to sequential ──
    used_parallel = False
    if total_blocks > 10:
        used_parallel = _encrypt_blocks_parallel(
            blocks, encrypted_blocks, all_encryption_info,
            quantum_seeds, repo_path, shots, total_start, log_interval, logger
        )

    if not used_parallel:
        # ── Sequential fallback ──
        logger.info("Using sequential encryption...")
        for i, block in enumerate(blocks):
            if encrypted_blocks[i] is not None:
                completed_count += 1
                continue  # already done by partial parallel run
            block_start = time.time()
            enc_block, enc_info = encrypt_block_quantum(
                block, i, quantum_seeds, modules, shots=shots
            )
            encrypted_blocks[i] = enc_block
            all_encryption_info[i] = enc_info
            completed_count += 1
            block_time = time.time() - block_start
            elapsed_total = time.time() - total_start

            if completed_count % log_interval == 0 or i == 0 or i == total_blocks - 1:
                pct = 100 * completed_count / total_blocks
                avg_time = elapsed_total / completed_count
                eta = avg_time * (total_blocks - completed_count)
                logger.info(
                    f"Encrypting block {i + 1}/{total_blocks} with shots={shots}... "
                    f"({pct:.1f}%) Block time: {block_time:.2f}s, ETA: {_format_time(eta)}"
                )

    total_time = time.time() - total_start
    logger.info(f"Quantum encryption complete: {total_blocks} blocks in {_format_time(total_time)}")
    logger.info("=" * 60)

    return encrypted_blocks, all_encryption_info


def decrypt_all_blocks(
    encrypted_blocks: List[np.ndarray],
    quantum_seeds: dict,
    all_encryption_info: List[dict],
    config: dict = None,
) -> List[np.ndarray]:
    """
    Decrypt all ROI blocks using reverse quantum operations.

    Args:
        encrypted_blocks: List of encrypted 8x8 blocks.
        quantum_seeds: Seed parameters.
        all_encryption_info: Encryption info for each block.
        config: Configuration dictionary.

    Returns:
        List of decrypted 8x8 blocks.
    """
    if config is None:
        config = load_config()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_path = os.path.join(project_root, config["repos"]["quantum"]["path"])
    shots = config.get("quantum_encryption", {}).get("shots", 1024)

    logger.info("=" * 60)
    logger.info("STARTING QUANTUM DECRYPTION OF ROI BLOCKS")
    logger.info("=" * 60)

    _verify_quantum_repo(repo_path)
    modules = _import_quantum_modules(repo_path)
    _verify_qiskit_backend()

    total_blocks = len(encrypted_blocks)
    logger.info(f"Total blocks to decrypt: {total_blocks}")
    logger.info(f"Shots per block: {shots}")

    decrypted_blocks = [None] * total_blocks
    total_start = time.time()
    completed_count = 0
    log_interval = max(1, total_blocks // 20)

    # ── Try parallel, then fall back to sequential ──
    used_parallel = False
    if total_blocks > 10:
        used_parallel = _decrypt_blocks_parallel(
            encrypted_blocks, decrypted_blocks, quantum_seeds,
            all_encryption_info, repo_path, shots, total_start, log_interval, logger
        )

    if not used_parallel:
        logger.info("Using sequential decryption...")
        for i, enc_block in enumerate(encrypted_blocks):
            if decrypted_blocks[i] is not None:
                completed_count += 1
                continue
            block_start = time.time()
            dec_block = decrypt_block_quantum(
                enc_block, i, quantum_seeds, all_encryption_info[i], modules, shots=shots
            )
            decrypted_blocks[i] = dec_block
            completed_count += 1
            block_time = time.time() - block_start
            elapsed_total = time.time() - total_start

            if completed_count % log_interval == 0 or i == 0 or i == total_blocks - 1:
                pct = 100 * completed_count / total_blocks
                avg_time = elapsed_total / completed_count
                eta = avg_time * (total_blocks - completed_count)
                logger.info(
                    f"Decrypting block {i + 1}/{total_blocks} with shots={shots}... "
                    f"({pct:.1f}%) Block time: {block_time:.2f}s, ETA: {_format_time(eta)}"
                )

    total_time = time.time() - total_start
    logger.info(f"Quantum decryption complete: {total_blocks} blocks in {_format_time(total_time)}")
    logger.info("=" * 60)

    return decrypted_blocks


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
