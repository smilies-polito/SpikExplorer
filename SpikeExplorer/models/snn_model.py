import torch
import snntorch as snn
from torch import nn
from snntorch import surrogate
from snntorch import utils


class SNN(nn.Module):
    input_size: int
    output_size: int
    hidden_list: list
    kernel_size: int

    def __init__(self, input_size, output_size, hidden_list: list, kernel_size, beta, device, surrogate_grad_type):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_list = hidden_list
        self.kernel_size = kernel_size
        self.beta = beta
        self.device = device
        self.surrogate_grad_type = surrogate_grad_type

    def make_snn(self):

        if self.surrogate_grad_type == "atan":
            # neuron and simulation parameters
            spike_grad = snn.surrogate.atan()
        else:
            spike_grad = snn.surrogate.sigmoid()

        try:
            output_first_seq = self.hidden_list.index(0)
        except ValueError:
            output_first_seq = self.output_size

        net = nn.Sequential(
            nn.Conv2d(self.input_size, output_first_seq, self.kernel_size),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2)
        )

        for n in range(len(self.hidden_list) - 1):
            net.append(
                nn.Conv2d(self.hidden_list.index(n), self.hidden_list.index(n + 1), self.kernel_size),
            )
            net.append(
                snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True)
            )
            net.append(
                nn.MaxPool2d(2)
            )

        net.append(
            nn.Flatten()
        )
        net.append(
            nn.Linear(
                self.hidden_list.index(len(self.hidden_list)) * self.kernel_size * self.kernel_size,
                self.output_size
            )
        )
        net.append(
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True, output=True)
        )
        net.to(self.device)

        return net


def forward_pass(net, data):
    spk_rec = []
    mem_rec = []
    snn.utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
