import pennylane as qml
import numpy as np

class AnsatzRepresentations:
    @staticmethod
    def hardware_efficient_ansatz(qubits, weights):
        """
        Generate the mathematical representation for Hardware Efficient Ansatz (HEA).

        Parameters:
        - qubits (int): Number of qubits.
        - weights (list): List of weights for the HEA circuit.

        Returns:
        - qml.QNode: Quantum node representing the HEA circuit.
        """
        dev = qml.device('default.qubit', wires=qubits)

        @qml.qnode(dev)
        def hea_circuit(weights):
            # Apply the BasicEntanglerLayers template
            qml.BasicEntanglerLayers(weights, wires=range(qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(qubits)]

        # Evaluate the HEA circuit with the provided weights
        return hea_circuit(weights)

    @staticmethod
    def alternating_layered_ansatz(qubits, initial_layer_weights, weights):
        """
        Generate the mathematical representation for Alternating Layered Ansatz (ALT).

        Parameters:
        - qubits (int): Number of qubits.
        - initial_layer_weights (list): List of weights for the initial layer.
        - weights (list): List of weights for the ALT circuit.

        Returns:
        - qml.QNode: Quantum node representing the ALT circuit.
        """
        dev = qml.device('default.qubit', wires=qubits)

        @qml.qnode(dev)
        def alt_circuit(initial_layer_weights, weights):
            # Apply the SimplifiedTwoDesign template
            qml.SimplifiedTwoDesign(initial_layer_weights=initial_layer_weights, weights=weights, wires=range(qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(qubits)]

        # Evaluate the ALT circuit with the provided weights
        return alt_circuit(initial_layer_weights, weights)

# Example usage
ansatz = AnsatzRepresentations()

# Example: provide random weights for HEA
weights_hea = np.random.normal(size=3)  # Adjust as needed
result_hea = ansatz.hardware_efficient_ansatz(qubits=3, weights=weights_hea)

# Example: provide random weights for ALT
init_weights_alt = np.random.normal(size=3)  # Adjust as needed
weights_layer1_alt = np.array([[0., np.pi], [0., np.pi]])
weights_layer2_alt = np.array([[np.pi, 0.], [np.pi, 0.]])
weights_alt = [weights_layer1_alt, weights_layer2_alt]
result_alt = ansatz.alternating_layered_ansatz(qubits=3, initial_layer_weights=init_weights_alt, weights=weights_alt)
