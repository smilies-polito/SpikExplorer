import torch
import snntorch as snn
from torch import nn
from snntorch import surrogate
from snntorch import utils
from logging import Logger
from ax.utils.common.logger import get_logger


logger: Logger = get_logger(__name__)


class SNN_NMNIST(nn.Module):
    input_size: int
    output_size: int
    hidden_list: list
    kernel_size: int

    def __init__(
        self,
        beta,
        device,
        surrogate_grad_type,
        active_consumption,
        idle_consumption,
        neuron_threshold,
    ):
        super().__init__()

        self.input_size = 2
        self.hidden_list = [12, 32]
        self.output_size = 10
        self.kernel_size = 5
        self.beta = beta
        self.device = device
        self.surrogate_grad_type = surrogate_grad_type
        self.active_consumption = active_consumption
        self.idle_consumption = idle_consumption
        self.neuron_threshold = neuron_threshold

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
            nn.Conv2d(
                self.input_size,
                output_first_seq,
                self.kernel_size,
            ),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
        )

        for n in range(len(self.hidden_list) - 1):
            net.extend(
                [
                    nn.Conv2d(
                        self.hidden_list[n],
                        self.hidden_list[n + 1],
                        self.kernel_size,
                    ),
                    snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
                    nn.MaxPool2d(2),
                ]
            )
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

    for step in range(10):  # data.size(0) = number of time steps
        #logger.info(data.size())
        #logger.info(data[step].size())
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
        if active_consumption and idle_consumption:
            for spike in spk_out:
                total_consumption += (
                    spike * active_consumption
                    if spike is 1
                    else spike * idle_consumption
                )

    if active_consumption and idle_consumption:
        return (
            torch.stack(spk_rec, dim=0),
            torch.stack(mem_rec, dim=0),
            total_consumption,
        )
    else:
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
