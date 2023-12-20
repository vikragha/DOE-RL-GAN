# utils/matrix_representations.py
import numpy as np
import pennylane as qml

class MatrixRepresentations:
    @staticmethod
    def ansatz_to_matrix(ansatz_representation):
        """
        Convert ansatz mathematical representation to matrix.

        Parameters:
        - ansatz_representation: An instance of the AnsatzRepresentations class.

        Returns:
        - np.ndarray: Matrix representation of the ansatz.
        """
        # Extract the QNode from the ansatz representation
        qnode = ansatz_representation()

        # Generate the matrix representation by evaluating the QNode
        matrix = qml.jacobian(qnode)(ansatz_representation.weights)
        return matrix

    @staticmethod
    def gates_to_matrix(gates_representation):
        """
        Convert gates mathematical representation to matrix.

        Parameters:
        - gates_representation: An instance of the GatesRepresentations class (if available).

        Returns:
        - np.ndarray: Matrix representation of the gates.
        """
        # Assuming gates_representation provides information about the gates used
        # Replace the following with the actual gates information from your representation
        gate_info = gates_representation.get_gate_info()

        # Implement conversion to matrix for each gate
        matrix_list = []
        for gate in gate_info:
            if gate == "RX":
                # Replace with actual parameters for RX gate
                matrix = rx_matrix(gate_info[gate])
            elif gate == "RY":
                # Replace with actual parameters for RY gate
                matrix = ry_matrix(gate_info[gate])
            elif gate == "RZ":
                # Replace with actual parameters for RZ gate
                matrix = rz_matrix(gate_info[gate])
            elif gate == "R":
                # Replace with actual parameters for R gate
                matrix = r_matrix(*gate_info[gate])
            else:
                raise ValueError(f"Unknown gate: {gate}")

            matrix_list.append(matrix)

        # Assuming the gates are applied sequentially
        final_matrix = np.eye(2 ** gates_representation.num_qubits)
        for matrix in matrix_list:
            final_matrix = np.dot(matrix, final_matrix)

        return final_matrix

# Define the CNOT gate matrix
cnot_matrix = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])

# Define the CZ gate matrix
cz_matrix = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]])

# Define RX gate matrix
def rx_matrix(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

# Define RY gate matrix
def ry_matrix(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])

# Define RZ gate matrix
def rz_matrix(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])

# Define R gate matrix
def r_matrix(alpha, beta, gamma):
    rz_matrix = np.array([[np.exp(-1j * (alpha + gamma) / 2), 0],
                          [0, np.exp(1j * (alpha + gamma) / 2)]])
    
    ry_matrix = np.array([[np.cos(beta / 2), -np.sin(beta / 2)],
                          [np.sin(beta / 2), np.cos(beta / 2)]])
    
    rz_minus_matrix = np.array([[np.exp(-1j * (alpha - gamma) / 2), 0],
                                [0, np.exp(1j * (alpha - gamma) / 2)]])
    
    return np.dot(rz_minus_matrix, np.dot(ry_matrix, rz_matrix))
