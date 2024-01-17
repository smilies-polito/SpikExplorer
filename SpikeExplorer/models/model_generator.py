import torch
import snntorch as snn
from torch import nn


class Net(nn.Module):
    """
    It will be created the Spiking Neural Network architecture here.
    input_size

    Args:
        input_size:
            type: int,
            defines the number of input neurons, depends on the dataset used
        hidden_layers:
            type: [int],
            defines how many hidden layer to insert with the respective size
        output_size:
            type: int,
            defines the number of neurons at the end of the net, it is strictly
            coupled to the dataset used
        beta:
            decay rate for snn layers
        time_steps:
            number of time_steps, it should be x.size(0) but
            it is given as input since we want to calibrate this parameter too
        neuron_type:
            type: str,
            defines which type of neuron will be used in the spiking layers
        network_type:
            type: str,
            defines if the network will be simple Feed-Forward, Convolutional, Recurrent
    """

    def __init__(self, input_size, hidden_layers: list, output_size, beta, time_steps, neuron_type="lif",
                 network_type=None, alpha=0.99, kernel_size=None):
        super().__init__()
        self.beta = beta
        self.time_steps = time_steps
        self.neuron_type = neuron_type
        self.network_type = network_type
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_num = len(hidden_layers)
        hidden_layers.append(output_size)

        if network_type == "conv":
            pass
        else:  # default case to ff
            self.__setattr__("fc1", nn.Linear(input_size, hidden_layers[0]))
            self._define_spiking_layer(
                beta=self.beta,
                hidden_layers=hidden_layers,
                counter=0,
                is_first=True,
                alpha=alpha,
                kernel_size=kernel_size
            )
            for counter in range(len(hidden_layers)):
                if counter != (len(hidden_layers) - 1):
                    self.__setattr__(f"fc{counter + 2}", nn.Linear(hidden_layers[counter], hidden_layers[counter + 1]))
                    self._define_spiking_layer(
                        beta=self.beta,
                        hidden_layers=hidden_layers,
                        counter=counter,
                        is_first=False,
                        alpha=alpha,
                        kernel_size=kernel_size
                    )

    def forward(self, x):
        if self.network_type == "conv":
            mem_list, spk_list = self._init_spiking_layers()

            # Record the final layer
            spk_rec = []
            mem_rec = []

            return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
        else:
            mem_list, spk_list = self._init_spiking_layers()

            # Record the final layer
            spk_rec = []
            mem_rec = []

            for step in range(self.time_steps):
                spk = x
                for num in range(self.hidden_layers_num + 1):
                    cur = self.__getattr__(f"fc{num + 1}")(spk)
                    spk, mem = self._spiking_forward_pass(num, cur, mem_list, spk_list)
                spk_rec.append(spk)
                mem_rec.append(mem)

            return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

    def _define_spiking_layer(self, beta, hidden_layers, counter, is_first, alpha=0.9, kernel_size=3):
        if is_first:
            if self.neuron_type == "lif":
                self.__setattr__(f"{self.neuron_type}1", snn.Leaky(beta=beta))
            elif self.neuron_type == "rlif":
                self.__setattr__(f"{self.neuron_type}1", snn.RLeaky(beta=beta, linear_features=hidden_layers[0]))
            elif self.neuron_type == "lap":
                self.__setattr__(f"{self.neuron_type}1", snn.Lapicque(beta=beta))
            elif self.neuron_type == "alp":
                self.__setattr__(f"{self.neuron_type}1", snn.Alpha(alpha=alpha, beta=beta))
            elif self.neuron_type == "syn":
                self.__setattr__(f"{self.neuron_type}1", snn.Synaptic(alpha=alpha, beta=beta))
            elif self.neuron_type == "rsyn":
                self.__setattr__(f"{self.neuron_type}1",
                                 snn.RSynaptic(alpha=alpha, beta=beta, linear_features=hidden_layers[0]))
            elif self.neuron_type == "syn":
                self.__setattr__(f"{self.neuron_type}1", snn.Synaptic(alpha=alpha, beta=beta))
            elif self.neuron_type == "SLSTM":
                self.__setattr__(f"{self.neuron_type}1", snn.SLSTM(input_size=hidden_layers[0],
                                                                   hidden_size=hidden_layers[0]))
            elif self.neuron_type == "SConv2dLSTM":
                self.__setattr__(f"{self.neuron_type}1", snn.SConv2dLSTM(in_channels=hidden_layers[0],
                                                                         out_channels=hidden_layers[0],
                                                                         kernel_size=kernel_size))
        else:
            if self.neuron_type == "lif":
                self.__setattr__(f"{self.neuron_type}{counter + 2}", snn.Leaky(beta=beta))
            elif self.neuron_type == "rlif":
                self.__setattr__(f"{self.neuron_type}{counter + 2}",
                                 snn.RLeaky(beta=beta, linear_features=hidden_layers[counter + 1]))
            elif self.neuron_type == "lap":
                self.__setattr__(f"{self.neuron_type}{counter + 2}", snn.Lapicque(beta=beta))
            elif self.neuron_type == "alp":
                self.__setattr__(f"{self.neuron_type}{counter + 2}", snn.Alpha(alpha=alpha, beta=beta))
            elif self.neuron_type == "syn":
                self.__setattr__(f"{self.neuron_type}{counter + 2}", snn.Synaptic(alpha=alpha, beta=beta))
            elif self.neuron_type == "rsyn":
                self.__setattr__(f"{self.neuron_type}{counter + 2}",
                                 snn.RSynaptic(alpha=alpha, beta=beta, linear_features=hidden_layers[counter + 1]))
            elif self.neuron_type == "syn":
                self.__setattr__(f"{self.neuron_type}{counter + 2}", snn.Synaptic(alpha=alpha, beta=beta))
            elif self.neuron_type == "SLSTM":
                self.__setattr__(f"{self.neuron_type}{counter + 2}", snn.SLSTM(input_size=hidden_layers[counter + 1],
                                                                               hidden_size=hidden_layers[counter + 1]))
            elif self.neuron_type == "SConv2dLSTM":
                self.__setattr__(f"{self.neuron_type}{counter + 2}",
                                 snn.SConv2dLSTM(in_channels=hidden_layers[counter + 1],
                                                 out_channels=hidden_layers[counter + 1],
                                                 kernel_size=kernel_size))

    def _init_spiking_layers(self):
        mem_list = []
        spk_list = []
        if self.neuron_type == "lif":
            mem_list.append(self.__getattr__(f"{self.neuron_type}1").init_leaky())
            for num in range(self.hidden_layers_num):
                mem_list.append(self.__getattr__(f"{self.neuron_type}{num + 1}").init_leaky())
        elif self.neuron_type == "rlif":
            spk_to_append, mem_to_append = self.__getattr__(f"{self.neuron_type}1").init_rleaky()
            spk_list.append(spk_to_append)
            mem_list.append(mem_to_append)
            for num in range(self.hidden_layers_num):
                spk_to_append, mem_to_append = self.__getattr__(f"{self.neuron_type}{num + 1}").init_rleaky()
                spk_list.append(spk_to_append)
                mem_list.append(mem_to_append)
        elif self.neuron_type == "lap":
            mem_list.append(self.__getattr__(f"{self.neuron_type}1").init_lapicque())
            for num in range(self.hidden_layers_num):
                mem_list.append(self.__getattr__(f"{self.neuron_type}{num + 1}").init_lapicque())
        elif self.neuron_type == "alp":
            mem_list.append(self.__getattr__(f"{self.neuron_type}1").init_alpha())
            for num in range(self.hidden_layers_num):
                syn_exc, syn_inh, mem = self.__getattr__(f"{self.neuron_type}{num + 1}").init_alpha()
                mem_list.append(mem)
        elif self.neuron_type == "syn":
            pass
        elif self.neuron_type == "rsyn":
            pass
        elif self.neuron_type == "syn":
            pass
        elif self.neuron_type == "SLSTM":
            pass
        elif self.neuron_type == "SConv2dLSTM":
            pass

        return mem_list, spk_list

    def _spiking_forward_pass(self, num, cur, mem_list, spk_list):
        spk, mem = None, None
        if self.neuron_type == "lif":
            spk, mem = self.__getattr__(f"{self.neuron_type}{num + 1}")(cur, mem_list[num])
        elif self.neuron_type == "rlif":
            spk, mem = self.__getattr__(f"{self.neuron_type}{num + 1}")(cur, spk_list[num], mem_list[num])
        elif self.neuron_type == "lap":
            spk, mem = self.__getattr__(f"{self.neuron_type}{num + 1}")(cur, mem_list[num])
        elif self.neuron_type == "alp":
            pass
        elif self.neuron_type == "syn":
            pass
        elif self.neuron_type == "rsyn":
            pass
        elif self.neuron_type == "syn":
            pass
        elif self.neuron_type == "SLSTM":
            pass
        elif self.neuron_type == "SConv2dLSTM":
            pass
        return spk, mem
