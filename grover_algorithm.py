import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit.primitives import Sampler
import matplotlib.pyplot as plt


def create_oracle(n_qubits, marked_state):
    oracle = QuantumCircuit(n_qubits)

    # Apply X gates to qubits that are 0 in the marked state
    for i, bit in enumerate(marked_state):
        if bit == '0':
            oracle.x(i)

    oracle.h(n_qubits - 1)
    oracle.mcp(np.pi, list(range(n_qubits - 1)), n_qubits - 1)
    oracle.h(n_qubits - 1)

    # Uncompute X gates
    for i, bit in enumerate(marked_state):
        if bit == '0':
            oracle.x(i)

    return oracle


def create_diffusion(n_qubits):
    diffusion = QuantumCircuit(n_qubits)

    # Apply H gates to all qubits
    for qubit in range(n_qubits):
        diffusion.h(qubit)

    # Apply X gates to all qubits
    for qubit in range(n_qubits):
        diffusion.x(qubit)

    # Apply multi-controlled Z gate
    diffusion.h(n_qubits - 1)
    diffusion.mcp(np.pi, list(range(n_qubits - 1)), n_qubits - 1)
    diffusion.h(n_qubits - 1)

    # Uncompute X gates
    for qubit in range(n_qubits):
        diffusion.x(qubit)

    # Uncompute H gates
    for qubit in range(n_qubits):
        diffusion.h(qubit)

    return diffusion


def grover_algorithm(n_qubits, marked_state, num_iterations):
    # Implements the actual Grover's Algo
    # Create quantum circuit
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initialize superposition
    for qubit in range(n_qubits):
        qc.h(qubit)

    # Apply Grover iterations
    for _ in range(num_iterations):
        # Apply oracle
        qc.append(create_oracle(n_qubits, marked_state), range(n_qubits))
        # Apply diffusion
        qc.append(create_diffusion(n_qubits), range(n_qubits))

    # Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


def main():
    # Parameters
    n_qubits = 3
    marked_state = '101'  # The state we want to find
    num_iterations = 2  # Number of Grover iterations

    # Creates the circuit
    qc = grover_algorithm(n_qubits, marked_state, num_iterations)

    # Execute the circuit using Sampler
    sampler = Sampler()
    job = sampler.run(qc, shots=1000)
    result = job.result()
    counts = result.quasi_dists[0]

    # Convert quasi_dists to regular counts
    counts_dict = {format(i, f'0{n_qubits}b'): int(prob * 1000) for i, prob in counts.items()}

    # Plot the results
    plt.figure(figsize=(10, 6))
    plot_histogram(counts_dict)
    plt.title(f"Grover's Algorithm Results (Marked State: {marked_state})")
    plt.savefig('grover_results.png')
    plt.close()


    # Just here to display the actual results
    print("\nResults:")
    print("--------")
    for state, count in sorted(counts_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"State {state}: {count} counts")

    print(f"\nThe marked state '{marked_state}' was found with probability: {counts_dict[marked_state] / 1000:.2%}")


if __name__ == "__main__":
    main()
