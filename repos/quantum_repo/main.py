

import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Project imports
# -----------------------------
# from quantum.mcqi import encode_mcqi, reconstruct_mcqi_image
from quantum.neqr import encode_neqr, reconstruct_neqr_image
from quantum.scrambling import (
    quantum_scramble,
    quantum_permutation,
    reverse_quantum_scrambling,
    reverse_quantum_permutation
)

from chaos.qrng import qrng
from chaos.henon import henon_map
from chaos.hybrid_map import generate_chaotic_key_image

from dna.dna_encode import dna_encode
from dna.dna_decode import dna_decrypt

from utils.metrics import compute_psnr_ssim


# -----------------------------
# Configuration
# -----------------------------
IMAGE_PATH = "data/Te-gl_0017.png"   # change if needed
IMG_SIZE = 64
SHOTS = 65536   # change to 4096 / 16384 / 65536
SEED_BITS = 16


# -----------------------------
# Load image
# -----------------------------
img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype(np.uint8)

print(f"[INFO] Image loaded: {img.shape}")


# -----------------------------
# Key generation
# -----------------------------
print("[INFO] Generating quantum-chaotic keys...")

x0 = qrng(SEED_BITS) / (2**SEED_BITS - 1)
y0 = qrng(SEED_BITS) / (2**SEED_BITS - 1)

x, y = henon_map(x0, y0, n_iter=IMG_SIZE)
bpk = np.floor(x * 256).astype(np.uint8)
ksk = np.floor(y * 256).astype(np.uint8)



# -----------------------------
# Quantum encryption
# -----------------------------
print("[INFO] Encoding image using MCQI...")
qc = encode_neqr(img)

num_position_qubits = int(2 * np.log2(IMG_SIZE))

print("[INFO] Applying quantum scrambling...")
qc = quantum_scramble(qc, bpk, num_position_qubits)

print("[INFO] Applying quantum permutation...")
qc = quantum_permutation(qc, ksk, num_position_qubits)

print("[INFO] Measuring quantum state...")
scrambled_quantum_img = reconstruct_neqr_image(
    qc, IMG_SIZE, IMG_SIZE, shots=SHOTS
)
# Resource counts

print("Total qubits:", qc.num_qubits)
print("Depth:", qc.depth())
print("Width:", qc.width())

# -----------------------------
# DNA encryption
# -----------------------------
print("[INFO] Applying DNA encryption...")
DNi0, DNi1, DNi2, DNi3 = dna_encode(scrambled_quantum_img, ksk)

KH = generate_chaotic_key_image(IMG_SIZE, IMG_SIZE)
DKi0, DKi1, DKi2, DKi3 = dna_encode(KH, ksk)

CMedi = (
    (DNi0 ^ DKi0) << 6 |
    (DNi1 ^ DKi1) << 4 |
    (DNi2 ^ DKi2) << 2 |
    (DNi3 ^ DKi3)
).astype(np.uint8)


# -----------------------------
# DNA decryption
# -----------------------------
print("[INFO] Decrypting DNA layer...")
scrambled_recovered = dna_decrypt(
    CMedi, DKi0, DKi1, DKi2, DKi3, ksk
)


# -----------------------------
# Reverse quantum operations
# -----------------------------
print("[INFO] Re-encoding recovered image...")
qc_re = encode_neqr(scrambled_quantum_img)

qc_re = reverse_quantum_permutation(qc_re, ksk, num_position_qubits)
qc_re = reverse_quantum_scrambling(qc_re, bpk, num_position_qubits)

print("[INFO] Final quantum reconstruction...")
reconstructed_img = reconstruct_neqr_image(
    qc_re, IMG_SIZE, IMG_SIZE, shots=SHOTS
)


# -----------------------------
# Evaluation
# -----------------------------
psnr_val, ssim_val = compute_psnr_ssim(img, reconstructed_img)

print("\n===== RESULTS =====")
print(f"Shots : {SHOTS}")
print(f"PSNR  : {psnr_val:.2f} dB")
print(f"SSIM  : {ssim_val:.4f}")


# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(reconstructed_img, cmap="gray")
plt.title(f"Reconstructed\nPSNR={psnr_val:.2f} dB | SSIM={ssim_val:.4f}")
plt.axis("off")

plt.tight_layout()
plt.show()
