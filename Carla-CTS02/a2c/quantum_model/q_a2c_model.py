import math
import time
import pennylane as qml
import torch
import torch.nn as nn
from a2c.classical_model.a2c_model import SharedNetwork, ActionNetwork, SharedNetworkScaled
import qiskit.providers.aer.noise as noise
import numpy as np
# from qiskit_aer import noise
from linetimer import CodeTimer
'''
There are three different circuit architectures which users can opt:
ansatz 1: Angle encoding  using Rx, Ry and Rz gates, followed by ansatz of Ry and Rz and Cz gates
ansatz 2: Angle encoding  using Rx, Ry and Rz gates, followed by ansatz of CRy gate
ansatz 3: Angle encoding + Ansatz1 combined to reduce the number of gates + Inputs scaled by trainable weights. 
'''
def _encode(n_qubits, inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[3 * wire + 0], wires=wire)
        qml.RY(inputs[3 * wire + 1], wires=wire)
        qml.RZ(inputs[3 * wire + 2], wires=wire)

def _encode_ansatz(n_qubits, inputs, inp_weights, y_params, z_params):
    for wire in range(n_qubits):
        qml.RX(inputs[3 * wire + 0]*inp_weights[wire][0], wires=wire)
        qml.RY(inputs[3 * wire + 1]*inp_weights[wire][1] + y_params[wire], wires=wire)
        qml.RZ(inputs[3 * wire + 2]*inp_weights[wire][2] + z_params[wire], wires=wire)

    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])


def _ansatz_1(n_qubits, y_weight, z_weight, gate_control_noise=False):
    if gate_control_noise:
        for wire, y_weight in enumerate(y_weight):
            qml.RY(y_weight+0.01*y_weight*np.random.rand(), wires=wire)
        for wire, z_weight in enumerate(z_weight):
            qml.RZ(z_weight+0.01*z_weight*np.random.rand(), wires=wire)
    else:
        for wire, y_weight in enumerate(y_weight):
            qml.RY(y_weight, wires=wire)
        for wire, z_weight in enumerate(z_weight):
            qml.RZ(z_weight, wires=wire)
    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])

    # qml.Barrier(only_visual =True)

def _get_noise_model():
    # For noise simulation use the following device
    # Error probabilities
    prob_1 = 0.001  # 1-qubit gate
    prob_2 = 0.01  # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz'])

    return noise_model


def _ansatz_2(n_qubits, y_weight):
    for wire, y_weight in enumerate(y_weight):
        qml.CRY(y_weight, wires=[wire, (wire + 1) % n_qubits])

def get_model(n_qubits, n_layers, num_sub_layer, ansatz, depolarising_error=True, gate_control_noise=True):

    if depolarising_error:
        dev = qml.device("qiskit.aer", wires=n_qubits, noise_model=_get_noise_model())
    else:
        dev = qml.device("default.qubit", wires=n_qubits)

    # dev = qml.device("qiskit.ibmq", wires=n_qubits, backend='ibm_lagos', shots=500)
                     # ibmqx_token="19aedf1abd199b48e90cb20f175c8901a8039a56f908d178c005a6b0495d22480846ad5adda61cd4c53038d0235ae68d240f77018b3a31794fb0401104ba8791")
    # print(qml.default_config)

    # To Visualise the circuit, use the following device
    # dev = qml.device("qiskit.aer", wires=n_qubits)

    data_per_qubit = 3
    if ansatz==1:
        shapes = {
            "y_weights": (n_layers, num_sub_layer, n_qubits),
            "z_weights": (n_layers, num_sub_layer, n_qubits),
        }
        @qml.qnode(dev, interface='torch')#, diff_method="parameter-shift") #, diff_method="backprop", shots= None)
        def circuit(inputs, y_weights, z_weights):

            def sublayer(inputs, sub_layer_idx, layer_idx, y_weights, z_weights):
                _encode(n_qubits, inputs[sub_layer_idx * data_per_qubit * n_qubits:(sub_layer_idx + 1) * data_per_qubit * n_qubits])
                _ansatz_1(n_qubits, y_weights[layer_idx][sub_layer_idx], z_weights[layer_idx][sub_layer_idx], gate_control_noise=gate_control_noise)


            for layer_idx in range(n_layers):
                for sub_layer_idx in range(num_sub_layer):
                    sublayer(inputs, sub_layer_idx, layer_idx, y_weights, z_weights)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        model = qml.qnn.TorchLayer(circuit, shapes)
        return model, dev
    elif ansatz==2:
        shapes = {
            "y_weights": (n_layers, num_sub_layer, n_qubits),
        }

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, y_weights):

            def sublayer(inputs, sub_layer_idx, layer_idx, y_weights):
                _encode(n_qubits, inputs[sub_layer_idx * data_per_qubit * n_qubits:(sub_layer_idx + 1) * data_per_qubit * n_qubits])
                _ansatz_2(n_qubits, y_weights[layer_idx][sub_layer_idx])

            for layer_idx in range(n_layers):
                for sub_layer_idx in range(num_sub_layer):
                    sublayer(inputs, sub_layer_idx, layer_idx, y_weights)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        model = qml.qnn.TorchLayer(circuit, shapes)
        return model, dev

    elif ansatz == 3:
        shapes = {
            "y_weights": (n_layers, num_sub_layer, n_qubits),
            "z_weights": (n_layers, num_sub_layer, n_qubits),
            "inp_weights": (n_layers, num_sub_layer, n_qubits, 3)
        }

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, y_weights, z_weights, inp_weights):

            def sublayer(inputs, sub_layer_idx, layer_idx, y_weights, z_weights, inp_weights):
                _encode_ansatz(n_qubits=n_qubits, inputs=inputs[sub_layer_idx * data_per_qubit * n_qubits:(sub_layer_idx + 1) * data_per_qubit * n_qubits],
                               inp_weights=inp_weights[layer_idx][sub_layer_idx], y_params=y_weights[layer_idx][sub_layer_idx], z_params=z_weights[layer_idx][sub_layer_idx])

            for layer_idx in range(n_layers):
                for sub_layer_idx in range(num_sub_layer):
                    sublayer(inputs, sub_layer_idx, layer_idx, y_weights, z_weights, inp_weights)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        # print(circuit)
        model = qml.qnn.TorchLayer(circuit, shapes)
        return model, dev


class QuantumNet(nn.Module):
    def __init__(self, n_layers, ansatz, num_sub_layer, n_qubits, depolarising_error, gate_control_noise):
        super(QuantumNet, self).__init__()
        self.n_qubits = n_qubits
        self.n_actions = 3

        self.q_layers, self.dev = get_model(n_qubits=self.n_qubits,
                                            n_layers=n_layers, num_sub_layer=num_sub_layer, ansatz=ansatz, depolarising_error=depolarising_error, gate_control_noise=gate_control_noise)

        # self.dev.tracker.active = True

        # print(self.q_layers)

        self.out_layer = nn.Linear(self.n_qubits, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = inputs.squeeze(0)
        inputs = torch.atan(inputs)
        outputs = self.q_layers(inputs)

        # # Printing statistics of job submitted to QC
        # job = self.dev._current_job
        # # Print the summary of job time per step
        # print('Summary: \n', self.dev.tracker.history)
        # # Print the time details for each step in running the job
        # print('Details: \n', job.time_per_step())

        # Following line of code draws the circuit
        # self.dev._circuit.draw(output="mpl", plot_barriers=True)

        # Output represents the output of the value function which by the RL problem design ranges between -1 and +1.
        # So, a tanh function can be applied after the final output.
        outputs = self.out_layer(outputs)
        # outputs = self.tanh(outputs)
        return outputs

class Q_A2C(nn.Module):
    def __init__(self, n_layers, ansatz, depolarising_error=False, gate_control_noise=False, n_qubits=4, dim_red=False, hidden_dim=32, num_actions=3, inp_scaling=False, model_eval=False):
        super().__init__()
        self.model_eval = model_eval
        if inp_scaling:
            self.shared_network = SharedNetworkScaled(hidden_dim=hidden_dim, dim_red=dim_red)
        else:
            self.shared_network = SharedNetwork(hidden_dim=hidden_dim, dim_red=dim_red)
        self.hidden_dim = hidden_dim
        self.bw_per_sublayer = n_qubits*3
        self.num_sub_layer = math.ceil(hidden_dim/self.bw_per_sublayer)
        # num_sub_layer = hidden_dim // (3 * n_qubits)
        self.action_policy = ActionNetwork(hidden_dim, num_actions)
        if not model_eval:
            self.value_network = QuantumNet(n_layers, ansatz, num_sub_layer=self.num_sub_layer, n_qubits=n_qubits, depolarising_error=depolarising_error, gate_control_noise = gate_control_noise)


    def forward(self, x, lstm_state, cat_tensor):
        x = x.permute(2, 0, 1)[None, :]
        value = None
        # torch.reshape(x, (-1, 3, 400, 400))
        cat_tensor = torch.reshape(cat_tensor, (-1, 4))
        obs = (x, lstm_state)
        # with CodeTimer("Shared Network"):
        features, cx = self.shared_network(obs, cat_tensor)
        t = time.time()
        if not self.model_eval:
            #Pads the hidden_dim vector
            if self.bw_per_sublayer*self.num_sub_layer != self.hidden_dim:
                #     Need to do input padding for value_network
                padding_len = self.bw_per_sublayer*self.num_sub_layer-self.hidden_dim
                # Cloned and detached the feature vector so that loss from value network is nor propagated backward to update shared_network
                features_value = torch.cat((features, torch.zeros((1,padding_len)).to(features.device)), dim=1)
                # with CodeTimer("Value Network"):
                value = self.value_network(features_value)
            else:
                value = self.value_network(features)
        # print("Time taken for quantum simulation: {:.4f}sec".format((time.time() - t)))
        # with CodeTimer("Actor Network"):
        action = self.action_policy(features)
        value = value.unsqueeze(0)
        return action, value, (features, cx)
