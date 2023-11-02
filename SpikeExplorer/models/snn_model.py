import torch
import snntorch as snn
from torch import nn
from snntorch import surrogate
from snntorch import utils
from logging import Logger
from ax.utils.common.logger import get_logger


logger: Logger = get_logger(__name__)


class SNN(nn.Module):
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
        self.input_size = 28*28
        self.hidden_size = 1000
        self.output_size = 10
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

        net = nn.Sequential(
            nn.Linear(
                self.input_size,
                self.hidden_size,
            ),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
        )
        net.extend(
            [
                nn.Linear(
                    self.hidden_size,
                    self.output_size,
                ),
                snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
            ]
        )
        net.to(self.device)

        return net


def forward_pass(net, data, active_consumption=None, idle_consumption=None):
    spk_rec = []
    mem_rec = []
    total_consumption = 0
    snn.utils.reset(net)  # resets hidden states for all LIF neurons in net
    spk_out, mem_out = net(data)
    spk_rec.append(spk_out)
    mem_rec.append(mem_out)
    if active_consumption and idle_consumption:
        for spike in spk_out:
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
