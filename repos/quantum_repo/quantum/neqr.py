import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate, MCXGate
from qiskit_aer import AerSimulator
from qiskit import transpile

# Module-level cached simulator instance (avoid recreating per-block)
_cached_simulator = None

def _get_simulator():
    global _cached_simulator
    if _cached_simulator is None:
        _cached_simulator = AerSimulator()
    return _cached_simulator


def encode_neqr(image):
    height, width = image.shape
    n = int(np.log2(height))

    num_position_qubits = 2 * n
    num_intensity_qubits = 8   # NEQR uses 8 bits for grayscale

    total_qubits = num_position_qubits + num_intensity_qubits
    qr = QuantumRegister(total_qubits, 'q')
    qc = QuantumCircuit(qr)

    # ─────────────────────────────────────────────
    # Step 1: Superposition of position qubits
    # ─────────────────────────────────────────────
    for i in range(num_position_qubits):
        qc.h(qr[i])

    # ─────────────────────────────────────────────
    # Step 2: NEQR encoding
    # ─────────────────────────────────────────────
    for i in range(height):
        for j in range(width):
            pixel_value = int(np.clip(image[i, j], 0, 255))
            if pixel_value == 0:
                continue

            # Binary position
            x_bin = format(i, f'0{n}b')
            y_bin = format(j, f'0{n}b')
            position_bin = x_bin + y_bin

            # Activate position
            for bit_pos, bit in enumerate(position_bin):
                if bit == '1':
                    qc.x(qr[bit_pos])

            # Binary intensity (8-bit)
            intensity_bin = format(pixel_value, '08b')

            controls = [qr[k] for k in range(num_position_qubits)]

            for bit_idx, bit in enumerate(intensity_bin[::-1]):  # LSB first
                if bit == '1':
                    target = qr[num_position_qubits + bit_idx]
                    mcx = MCXGate(len(controls))
                    qc.append(mcx, controls + [target])

            # Reset position
            for bit_pos, bit in enumerate(position_bin):
                if bit == '1':
                    qc.x(qr[bit_pos])

    return qc


def reconstruct_neqr_image(qc, height, width, shots=8192):

    qc_measured = qc.copy()
    qc_measured.measure_all()

    simulator = _get_simulator()
    compiled = transpile(qc_measured, simulator, optimization_level=0)
    job = simulator.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    n = int(np.log2(height))
    num_position_qubits = 2 * n
    num_intensity_qubits = 8

    recon_img = np.zeros((height, width), dtype=np.uint8)

    for bitstring, count in counts.items():

        bitstring = bitstring[::-1]   # endian fix (same as MCQI)

        # ── Split position and intensity ──
        pos_bits = bitstring[:num_position_qubits]
        intensity_bits = bitstring[
            num_position_qubits : num_position_qubits + num_intensity_qubits
        ]

        i = int(pos_bits[:n], 2)
        j = int(pos_bits[n:], 2)

        if i < height and j < width:
            intensity_val = int(intensity_bits[::-1], 2)  # LSB first
            recon_img[i, j] = intensity_val

    return recon_img