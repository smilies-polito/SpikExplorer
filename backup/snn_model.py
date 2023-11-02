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

    def __init__(
        self,
        input_size,
        output_size,
        hidden_list: list,
        kernel_size,
        beta,
        device,
        surrogate_grad_type,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_list = hidden_list
        self.kernel_size = kernel_size
        self.beta = beta
        self.device = device
        self.surrogate_grad_type = surrogate_grad_type
        self.active_consumption = 0.5
        self.idle_consumption = 0.1

    def make_snn(self):
        if self.surrogate_grad_type == "atan":
            # neuron and simulation parameters
            spike_grad = snn.surrogate.atan()
        else:
            spike_grad = snn.surrogate.sigmoid()

        try:
            output_first_seq = self.hidden_list[0]
        except ValueError:
            output_first_seq = self.output_size

        net = nn.Sequential(
            nn.Conv2d(self.input_size, output_first_seq, self.kernel_size),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
        )

        for n in range(len(self.hidden_list) - 1):
            net.append(
                nn.Conv2d(
                    self.hidden_list[n],
                    self.hidden_list[n + 1],
                    self.kernel_size,
                ),
            )
            net.append(
                snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True)
            )
            net.append(nn.MaxPool2d(2))

        net.append(nn.Flatten())
        net.append(
            nn.Linear(
                self.hidden_list[len(self.hidden_list) - 1]
                * self.kernel_size
                * self.kernel_size,
                self.output_size,
            )
        )
        net.append(
            snn.Leaky(
                beta=self.beta, spike_grad=spike_grad, init_hidden=True, output=True
            )
        )
        net.to(self.device)

        return net


def forward_pass(net, data, active_consumption=None, idle_consumption=None):
    spk_rec = []
    mem_rec = []
    total_consumption = 0
    snn.utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(20):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
        if active_consumption and idle_consumption:
            for spikes in spk_out:
                for spike in spikes:
                    total_consumption += (
                        active_consumption
                        if spike == 1
                        else idle_consumption
                    )

    if active_consumption and idle_consumption:
        return (
            torch.stack(spk_rec, dim=0),
            torch.stack(mem_rec, dim=0),
            total_consumption,
        )
    else:
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
