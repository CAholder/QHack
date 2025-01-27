#! /usr/bin/python3

import sys

import numpy
from pennylane import numpy as np
import pennylane as qml


def distance(A, B):
    """Function that returns the distance between two vectors.

    Args:
        - A (list[int]): person's information: [age, minutes spent watching TV].
        - B (list[int]): person's information: [age, minutes spent watching TV].

    Returns:
        - (float): distance between the two feature vectors.
    """

    # QHACK #

    # The Swap test is a method that allows you to calculate |<A|B>|^2 , you could use it to help you.
    # The qml.AmplitudeEmbedding operator could help you too.
    # Supposed to train on Qubit 1 with no labels
    # Supposed to train on Qubit 2 with labels
    num_wires = int(np.ceil(np.log2(len(A))))

    dev = qml.device("default.qubit", wires=num_wires*3)

    @qml.qnode(dev)
    def reg1(inputs_valid):
        qml.AmplitudeEmbedding(features=inputs_valid[2:4], wires=range(num_wires, num_wires * 3), normalize=True,
                               pad_with=0.)
        ancillea = []

        # for i in range(num_wires):
        #
        #     anc = i
        #     ancillea.append(anc)
        #     first_qubit = i+num_wires
        #     second_qubit = i + (2*num_wires)
        #
        #     # qml.PauliX(wires=first_qubit)
        #     qml.PauliX(wires=second_qubit)
        #     qml.Identity(wires=second_qubit)
        #     qml.Hadamard(wires=anc)
        #     qml.CSWAP(wires=[anc, first_qubit, second_qubit])
        #     qml.Hadamard(wires=anc)
        anc = 0
        qml.Hadamard(wires=anc)
        # qml.RX(inputs_valid[], wires=1)
        # qml.RX(inputs_valid[1], wires=2)
        qml.CSWAP(wires=[anc, 1, 2])
        qml.Hadamard(wires=anc)



        # for i in range(num_wires):
        #     anc = i
        #     ancillea.append(anc)
        #     qml.Hadamard(wires=anc)
        #     for x in range(num_wires):
        #         first_state = x + num_wires
        #         # qml.PauliX(wires=first_state)
        #         second_state = x + 2*num_wires
        #         # qml.PauliX(wires=second_state)
        #         qml.CSWAP(wires=[anc,first_state,second_state])
        #     qml.Hadamard(wires=anc)

        # return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillea]))
        return qml.expval(qml.PauliZ(anc))
        # return qml.expval(*[qml.PauliZ(i) for i in ancillea])
    inputs_valid = np.concatenate([A, B])
    print("This is A", A)
    print("This is B", B)
    print(inputs_valid)
    print(inputs_valid[2:4])
    result = reg1(inputs_valid)
    # print(np.sqrt(result))
    # np.sqrt(np.abs(result))
    # np.sqrt(2*(1-(np.abs(result))))
    result2 = np.sqrt(result)
    return np.sqrt(2*(1-(np.abs(result2))))
    # QHACK #


def predict(dataset, new, k):
    """Function that given a dataset, determines if a new person do like Beatles or not.

    Args:
        - dataset (list): List with the age, minutes that different people watch TV, and if they like Beatles.
        - new (list(int)): Age and TV minutes of the person we want to classify.
        - k (int): number of nearby neighbors to be taken into account.

    Returns:
        - (str): "YES" if they like Beatles, "NO" otherwise.
    """

    # DO NOT MODIFY anything in this code block

    def k_nearest_classes():
        """Function that returns a list of k near neighbors."""
        distances = []
        for data in dataset:
            distances.append(distance(data[0], new))
        nearest = []
        for _ in range(k):
            indx = np.argmin(distances)
            nearest.append(indx)
            distances[indx] += 2

        return [dataset[i][1] for i in nearest]

    output = k_nearest_classes()

    return (
        "YES" if len([i for i in output if i == "YES"]) > len(output) / 2 else "NO",
        float(distance(dataset[0][0], new)),
    )


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    dataset = []
    new = [int(inputs[0]), int(inputs[1])]
    k = int(inputs[2])
    for i in range(3, len(inputs), 3):
        dataset.append([[int(inputs[i + 0]), int(inputs[i + 1])], str(inputs[i + 2])])

    output = predict(dataset, new, k)
    sol = 0 if output[0] == "YES" else 1
    print(f"{sol},{output[1]}")
