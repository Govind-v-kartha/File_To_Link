import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Module-level cached simulator
_cached_simulator = None

def _get_simulator():
    global _cached_simulator
    if _cached_simulator is None:
        _cached_simulator = AerSimulator()
    return _cached_simulator

def qrng(num_bits):
    qc = QuantumCircuit(num_bits, num_bits)
    qc.h(range(num_bits))
    qc.measure(range(num_bits), range(num_bits))

    simulator = _get_simulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result()
    counts = result.get_counts(qc)

    return int(list(counts.keys())[0], 2)