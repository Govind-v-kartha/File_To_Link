import numpy as np
from chaos.qrng import qrng
def hybrid_logistic_sine_map(x0, r, iterations):
    x = x0
    seq = np.zeros(iterations, dtype=np.float64)

    for i in range(iterations):
        x = r * x * (1 - x) + 4 * r * np.sin(np.pi * x / 4)
        x = x % 1.0
        seq[i] = x

    return seq


def generate_chaotic_key_image(height, width):
    total_pixels = height * width

    r = qrng(16) / 65535.0
    x0 = qrng(16) / 65535.0

    seq = hybrid_logistic_sine_map(x0, r, total_pixels)

    KH = np.floor(seq * 256).astype(np.uint8)
    KH = KH.reshape((height, width))

    return KH