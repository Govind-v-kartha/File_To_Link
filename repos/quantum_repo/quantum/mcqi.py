import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np

def encode_mcqi(image):
    height, width = image.shape
    n = int(np.log2(height))
    num_position_qubits = 2 * n
    num_intensity_qubits = 1

    total_qubits = num_position_qubits + num_intensity_qubits
    qr = QuantumRegister(total_qubits, 'q')
    qc = QuantumCircuit(qr)


    for i in range(num_position_qubits):
        qc.h(qr[i])

    for i in range(height):
        for j in range(width):
            pixel_value = np.clip(image[i, j], 0, 255) / 255.0
            if pixel_value == 0:
                continue

            #theta = np.pi * pixel_value
            pixel_value = image[i, j] / 255.0
            theta = np.pi * pixel_value


            x_bin = format(i, f'0{n}b')
            y_bin = format(j, f'0{n}b')
            position_bin = x_bin + y_bin

            for bit_pos, bit in enumerate(position_bin):
                if bit == '1':
                    qc.x(qr[bit_pos])

            control_qubits = [qr[k] for k in range(num_position_qubits)]
            target_qubit = qr[num_position_qubits]
            cry_gate = RYGate(theta).control(len(control_qubits))
            qc.append(cry_gate, control_qubits + [target_qubit])

            for bit_pos, bit in enumerate(position_bin):
                if bit == '1':
                    qc.x(qr[bit_pos])

    return qc

# qc_mcqi = encode_mcqi(img)
# qc_mcqi.draw()


def reconstruct_mcqi_image(qc, height, width, shots=9192):

    qc_measured = qc.copy()
    qc_measured.measure_all()

    simulator = AerSimulator()
    compiled = transpile(qc_measured, simulator)
    job = simulator.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    n = int(np.log2(height))
    num_position_qubits = 2 * n

    pixel_one_counts = np.zeros((height, width), dtype=np.float64)
    pixel_total_counts = np.zeros((height, width), dtype=np.float64)

    for bitstring, count in counts.items():

        bitstring = bitstring[::-1]   # endian fix

        pos_bits = bitstring[:num_position_qubits]
        intensity_bit = bitstring[num_position_qubits]

        i = int(pos_bits[:n], 2)
        j = int(pos_bits[n:], 2)

        if i < height and j < width:
            pixel_total_counts[i, j] += count
            if intensity_bit == '1':
                pixel_one_counts[i, j] += count

    scrambled_img = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if pixel_total_counts[i, j] > 0:
                p = pixel_one_counts[i, j] / pixel_total_counts[i, j]
                scrambled_img[i, j] = int(p * 255)   # <-- correct line

    return scrambled_img



# # ===== usage =====
# permuted_image = reconstruct_mcqi_image(qc_permuted, 64, 64)

# plt.imshow(permuted_image, cmap='gray')
# plt.title("Permuted Image (Quantum reconstructed)")
# plt.axis('off')
# plt.show()