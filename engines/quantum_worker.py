"""
Standalone quantum block worker for multiprocessing.

This module is designed to be imported by child processes spawned via
ProcessPoolExecutor on Windows. It does NOT import from utils.* or any
project-specific modules at the top level — those are imported lazily
inside the worker functions to avoid import failures in child processes.
"""

import os
import sys
import time
import warnings
import numpy as np

# Suppress Henon map NaN/Inf cast and overflow warnings in child processes
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Module-level cache for quantum modules (populated once per process)
_modules = None
_repo_path = None


def worker_initializer(repo_path: str):
    """Pre-load quantum modules when the worker process starts.

    Called once per worker by ProcessPoolExecutor(initializer=...).
    This ensures the heavy Qiskit/Aer imports happen before any task
    is dispatched, reducing the chance of a simultaneous OOM spike
    when all workers import at the same time.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    _ensure_modules(repo_path)


def _ensure_modules(repo_path: str) -> dict:
    """Import quantum repo modules (cached per process)."""
    global _modules, _repo_path
    if _modules is not None and _repo_path == repo_path:
        return _modules

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

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

    _modules = {
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
    _repo_path = repo_path
    return _modules


def _generate_keys_for_block(quantum_seeds, block_id, block_size, modules):
    """Generate chaotic keys for a single block."""
    henon_map = modules["henon_map"]
    x0 = quantum_seeds["x0"]
    y0 = quantum_seeds["y0"]
    alpha = quantum_seeds.get("alpha", 1.8)
    beta = quantum_seeds.get("beta", 0.015)

    seed_x = (x0 + block_id * 0.001) % 1.0
    seed_y = (y0 + block_id * 0.0007) % 1.0

    iterations = block_size * block_size + 100
    x, y = henon_map(seed_x, seed_y, alpha, beta, iterations)

    x = x[100:]
    y = y[100:]

    x = np.nan_to_num(x, nan=0.5, posinf=0.99, neginf=0.01)
    y = np.nan_to_num(y, nan=0.5, posinf=0.99, neginf=0.01)

    bpk = np.floor(np.abs(x) * 256).astype(np.uint8)
    ksk = np.floor(np.abs(y) * 256).astype(np.uint8)

    return bpk, ksk


def _rgb_to_grayscale(block):
    """Convert RGB block to grayscale."""
    if block.ndim == 3:
        return np.dot(block[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return block.copy()


def encrypt_block_worker(args):
    """
    Encrypt a single 8x8 block. Designed for use with ProcessPoolExecutor.

    Args:
        args: Tuple of (block, block_id, quantum_seeds, repo_path, shots)

    Returns:
        Tuple of (block_id, encrypted_block, encryption_info)
    """
    block, block_id, quantum_seeds, repo_path, shots = args
    try:
        modules = _ensure_modules(repo_path)

        block_size = 32
        encode_neqr = modules["encode_neqr"]
        reconstruct_neqr_image = modules["reconstruct_neqr_image"]
        q_scramble = modules["quantum_scramble"]
        q_permutation = modules["quantum_permutation"]
        dna_encode = modules["dna_encode"]

        gray_block = _rgb_to_grayscale(block)
        bpk, ksk = _generate_keys_for_block(quantum_seeds, block_id, block_size, modules)

        qc = encode_neqr(gray_block)
        n = int(np.log2(block_size))
        num_position_qubits = 2 * n

        qc = q_scramble(qc, bpk, num_position_qubits)
        qc = q_permutation(qc, ksk, num_position_qubits)
        scrambled_img = reconstruct_neqr_image(qc, block_size, block_size, shots=shots)

        DNi0, DNi1, DNi2, DNi3 = dna_encode(scrambled_img, ksk)

        np.random.seed(int(bpk.sum()) * 1000 + int(ksk.sum()) + block_id)
        KH = np.random.randint(0, 256, (block_size, block_size), dtype=np.uint8)
        DKi0, DKi1, DKi2, DKi3 = dna_encode(KH, ksk)

        encrypted_block = (
            (DNi0 ^ DKi0) << 6
            | (DNi1 ^ DKi1) << 4
            | (DNi2 ^ DKi2) << 2
            | (DNi3 ^ DKi3)
        ).astype(np.uint8)

        encryption_info = {
            "block_id": block_id,
            "shots": shots,
            "num_qubits": qc.num_qubits,
            "circuit_depth": qc.depth(),
            "bpk": bpk.tolist(),
            "ksk": ksk.tolist(),
            "KH": KH.tolist(),
            "DKi0": DKi0.tolist(),
            "DKi1": DKi1.tolist(),
            "DKi2": DKi2.tolist(),
            "DKi3": DKi3.tolist(),
        }

        return block_id, encrypted_block, encryption_info
    except Exception as e:
        raise RuntimeError(f"encrypt_block_worker block {block_id}: {e}") from e


def decrypt_block_worker(args):
    """
    Decrypt a single 8x8 block. Designed for use with ProcessPoolExecutor.

    Args:
        args: Tuple of (encrypted_block, block_id, quantum_seeds, encryption_info, repo_path, shots)

    Returns:
        Tuple of (block_id, decrypted_block)
    """
    encrypted_block, block_id, quantum_seeds, encryption_info, repo_path, shots = args
    try:
        modules = _ensure_modules(repo_path)

        block_size = 32
        encode_neqr = modules["encode_neqr"]
        reconstruct_neqr_image = modules["reconstruct_neqr_image"]
        reverse_q_scrambling = modules["reverse_quantum_scrambling"]
        reverse_q_permutation = modules["reverse_quantum_permutation"]
        dna_decrypt = modules["dna_decrypt"]

        bpk = np.array(encryption_info["bpk"], dtype=np.uint8)
        ksk = np.array(encryption_info["ksk"], dtype=np.uint8)

        DKi0 = np.array(encryption_info["DKi0"], dtype=np.uint8)
        DKi1 = np.array(encryption_info["DKi1"], dtype=np.uint8)
        DKi2 = np.array(encryption_info["DKi2"], dtype=np.uint8)
        DKi3 = np.array(encryption_info["DKi3"], dtype=np.uint8)

        n = int(np.log2(block_size))
        num_position_qubits = 2 * n

        scrambled_recovered = dna_decrypt(encrypted_block, DKi0, DKi1, DKi2, DKi3, ksk)
        qc_re = encode_neqr(scrambled_recovered)
        qc_re = reverse_q_permutation(qc_re, ksk, num_position_qubits)
        qc_re = reverse_q_scrambling(qc_re, bpk, num_position_qubits)
        decrypted_block = reconstruct_neqr_image(qc_re, block_size, block_size, shots=shots)

        return block_id, decrypted_block
    except Exception as e:
        raise RuntimeError(f"decrypt_block_worker block {block_id}: {e}") from e
