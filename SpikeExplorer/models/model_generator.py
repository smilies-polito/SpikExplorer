import torch
import snntorch as snn
from snntorch import spikegen
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

    def __init__(self, input_size, hidden_layers: list, output_size, beta, time_steps, neuron_type=None,
                 network_type=None, alpha=0.99, kernel_size=None, dataset="mnist", learnable_exp_decay=False,
                 active_consumption=1, passive_consumtpion=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.time_steps = time_steps
        self.neuron_type = neuron_type
        self.network_type = network_type
        self.active_consumption = active_consumption
        self.passive_consumption = passive_consumtpion
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_plus_out_layers_num = len(hidden_layers)
        self.hidden_plus_out_layers = hidden_layers.copy()
        self.hidden_plus_out_layers.append(output_size)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dataset = dataset
        self.learnable_exp_decay = learnable_exp_decay

        if network_type == "conv":
            pass
        else:  # default case to ff
            if self.dataset == "dvs":
                self.input_size = 16384
            self.__setattr__("fc1", nn.Linear(self.input_size, self.hidden_plus_out_layers[0]))
            self._define_spiking_layer(
                beta=self.beta,
                hidden_layers=self.hidden_plus_out_layers,
                counter=0,
                is_first=True,
                alpha=self.alpha,
                kernel_size=kernel_size
            )
            for counter in range(len(self.hidden_plus_out_layers)):
                if counter != (len(self.hidden_plus_out_layers) - 1):
                    self.__setattr__(f"fc{counter + 2}", nn.Linear(
                        self.hidden_plus_out_layers[counter], self.hidden_plus_out_layers[counter + 1]))
                    self._define_spiking_layer(
                        beta=self.beta,
                        hidden_layers=self.hidden_plus_out_layers,
                        counter=counter,
                        is_first=False,
                        alpha=self.alpha,
                        kernel_size=kernel_size
                    )

    def forward(self, x):
        if self.network_type == "conv":
            mem_list, spk_list, _ = self._init_spiking_layers()

            # Record the final layer
            spk_rec = []
            mem_rec = []

            return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
        else:
            mem = None
            syn_list = []
            if self.neuron_type in ["syn", "rsyn"]:
                mem_list, spk_list, syn_list = self._init_spiking_layers()
            else:
                mem_list, spk_list, _ = self._init_spiking_layers()

            # Record the final layer
            spk_rec = []
            mem_rec = []

            spk_dict = {}
            mem_dict = {}

            if self.dataset == "mnist":
                input_spikes = spikegen.rate(x, num_steps=self.time_steps,
                                         gain=1)

            for step in range(self.time_steps):
                if self.dataset == "mnist":
                    spk = input_spikes[step]
                elif self.dataset == "dvs":
                    spk = x[step,:,:]
                elif self.dataset == "shd":
                    spk = x[:,step,:]
                spk = spk.float()
                for num in range(self.hidden_plus_out_layers_num + 1):
                    cur = self.__getattr__(f"fc{num + 1}")(spk)
                    if self.neuron_type in ["syn", "rsyn"]:
                        spk, mem, syn = self._spiking_forward_pass(num, cur, mem_list, spk_list, syn_list)
                    else:
                        spk, mem, _ = self._spiking_forward_pass(num, cur, mem_list, spk_list)
                    if not spk_dict.get(f"{self.neuron_type}{num + 1}"):
                        spk_dict[f"{self.neuron_type}{num + 1}"] = []
                    if not mem_dict.get(f"{self.neuron_type}{num + 1}"):
                        mem_dict[f"{self.neuron_type}{num + 1}"] = []
                    spk_dict[f"{self.neuron_type}{num + 1}"].append(spk)
                    mem_dict[f"{self.neuron_type}{num + 1}"].append(mem)
                spk_rec.append(spk)
                mem_rec.append(mem)

            total_consumption = self._calculate_total_consumption(spk_dict, mem_dict)

            return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0), total_consumption

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
        syn_list = []
        if self.neuron_type == "lif":
            mem_list.append(self.__getattr__(f"{self.neuron_type}1").init_leaky())
            for num in range(self.hidden_plus_out_layers_num):
                mem_list.append(self.__getattr__(f"{self.neuron_type}{num + 1}").init_leaky())
        elif self.neuron_type == "rlif":
            spk_to_append, mem_to_append = self.__getattr__(f"{self.neuron_type}1").init_rleaky()
            spk_list.append(spk_to_append)
            mem_list.append(mem_to_append)
            for num in range(self.hidden_plus_out_layers_num):
                spk_to_append, mem_to_append = self.__getattr__(f"{self.neuron_type}{num + 1}").init_rleaky()
                spk_list.append(spk_to_append)
                mem_list.append(mem_to_append)
        elif self.neuron_type == "lap":
            mem_list.append(self.__getattr__(f"{self.neuron_type}1").init_lapicque())
            for num in range(self.hidden_plus_out_layers_num):
                mem_list.append(self.__getattr__(f"{self.neuron_type}{num + 1}").init_lapicque())
        elif self.neuron_type == "alp":
            mem_list.append(self.__getattr__(f"{self.neuron_type}1").init_alpha())
            for num in range(self.hidden_plus_out_layers_num):
                syn_exc, syn_inh, mem = self.__getattr__(f"{self.neuron_type}{num + 1}").init_alpha()
                mem_list.append(mem)
        elif self.neuron_type == "syn":
            syn, mem = self.__getattr__(f"{self.neuron_type}1").init_synaptic()
            syn_list.append(syn)
            mem_list.append(mem)
            for num in range(self.hidden_plus_out_layers_num):
                syn, mem = self.__getattr__(f"{self.neuron_type}{num + 1}").init_synaptic()
                syn_list.append(syn)
                mem_list.append(mem)
        elif self.neuron_type == "rsyn":
            spk, syn, mem = self.__getattr__(f"{self.neuron_type}1").init_rsynaptic()
            spk_list.append(spk)
            syn_list.append(syn)
            mem_list.append(mem)
            for num in range(self.hidden_plus_out_layers_num):
                spk, syn, mem = self.__getattr__(f"{self.neuron_type}{num + 1}").init_rsynaptic()
                spk_list.append(spk)
                syn_list.append(syn)
                mem_list.append(mem)
        elif self.neuron_type == "SLSTM":
            pass
        elif self.neuron_type == "SConv2dLSTM":
            pass

        return mem_list, spk_list, syn_list

    def _spiking_forward_pass(self, num, cur, mem_list, spk_list, syn_list=None):
        spk, mem, syn = None, None, None
        if self.neuron_type == "lif":
            spk, mem = self.__getattr__(f"{self.neuron_type}{num + 1}")(cur, mem_list[num])
        elif self.neuron_type == "rlif":
            spk, mem = self.__getattr__(f"{self.neuron_type}{num + 1}")(cur, spk_list[num], mem_list[num])
        elif self.neuron_type == "lap":
            spk, mem = self.__getattr__(f"{self.neuron_type}{num + 1}")(cur, mem_list[num])
        elif self.neuron_type == "alp":
            pass
        elif self.neuron_type == "syn":
            spk, syn, mem = self.__getattr__(f"{self.neuron_type}{num + 1}")(cur, syn_list[num], mem_list[num])
        elif self.neuron_type == "rsyn":
            spk, syn, mem = self.__getattr__(f"{self.neuron_type}{num + 1}")(cur, spk_list[num], syn_list[num], mem_list[num])
        elif self.neuron_type == "syn":
            pass
        elif self.neuron_type == "SLSTM":
            pass
        elif self.neuron_type == "SConv2dLSTM":
            pass
        return spk, mem, syn

    def _calculate_total_consumption(self, spk_dict: dict, mem_dict: dict):
        total_consumption = 0
        for layer in mem_dict.keys():
            # layer_spk_tensor = torch.stack(spk_dict[layer], dim=0)     # time_steps x batch_size x num_neurons
            layer_mem_tensor = torch.stack(mem_dict[layer], dim=0)     # same dimensions as above
            delta_mem = layer_mem_tensor[1:-1]-layer_mem_tensor[0:-2]
            total_consumption += torch.sum(torch.where((delta_mem > 0), 0.3, 0.1))
        return total_consumption.__float__()
