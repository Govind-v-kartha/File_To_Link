def quantum_scramble(qc, bpk, num_position_qubits):
    for i in range(num_position_qubits):
        key = bpk[i % len(bpk)]
        if key % 2 == 1:
            qc.x(i)
        if (key // 2) % 2 == 1:
            qc.z(i)
    return qc

# num_position_qubits = 12
# qc_scrambled = quantum_scramble(qc_mcqi, bpk, num_position_qubits)
# qc_scrambled.draw()
def quantum_permutation(qc, ksk, num_position_qubits):
    swap_operations = []
    for i in range(num_position_qubits - 1):
        j = i + 1 + (ksk[i % len(ksk)] % (num_position_qubits - i - 1))
        j = min(j, num_position_qubits - 1)
        swap_operations.append((i, j))

    for i, j in swap_operations:
        qc.swap(i, j)

    return qc

def reverse_quantum_permutation(qc_permuted, ksk, num_position_qubits):
    qc_unpermuted = qc_permuted.copy()
    swap_operations=[]
    for i in range(num_position_qubits - 1):
        j=i+1+(ksk[i%len(ksk)]%(num_position_qubits-i-1))
        j=min(j,num_position_qubits-1)
        swap_operations.append((i,j))
    for i,j in reversed(swap_operations):
        qc_unpermuted.swap(i,j)
    return qc_unpermuted

def reverse_quantum_scrambling(qc_scrambled, bpk, num_position_qubits):
    qc_unscrambled = qc_scrambled.copy()
    for i in range(num_position_qubits):
        key=bpk[i%len(bpk)]
        if (key//2)%2==1: qc_unscrambled.z(i)
        if key%2==1: qc_unscrambled.x(i)
    return qc_unscrambled

# qc_unpermuted = reverse_quantum_permutation(qc_permuted, ksk, 12)
# qc_original_state = reverse_quantum_scrambling(qc_unpermuted, bpk, 12)