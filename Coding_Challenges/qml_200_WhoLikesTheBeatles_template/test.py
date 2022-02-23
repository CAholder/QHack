import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding

# define two random input vectors
inp1 = np.array([1, 2, 0, 0])
inp2 = np.array([0, 1, 0, 0])

qubit_number = int(np.ceil(np.log2(len(inp1))))
print(qubit_number)

dev = qml.device('default.qubit', wires=3 * qubit_number)


@qml.qnode(dev)
def prepare_reg1(inp):
    AmplitudeEmbedding(features=inp, wires=range(qubit_number, 3 * qubit_number), normalize=True, pad_with=0.)

    ancillea = []
    for i in range(qubit_number):
        anc = i
        ancillea.append(anc)
        first_state = i + qubit_number
        second_state = i + 2 * qubit_number
        qml.CSWAP(wires=[anc, first_state, second_state])

    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillea]))


input_state = np.concatenate([inp1, inp2])
print(input_state)
prepare_reg1(input_state)

# print(prepare_reg1.draw())