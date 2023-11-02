import torch
import snntorch as snn
from torch import nn
from snntorch import surrogate
from snntorch import utils
from logging import Logger
from ax.utils.common.logger import get_logger


logger: Logger = get_logger(__name__)


class SNN_SHD(nn.Module):
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
        stride=None,
        padding=None,
    ):
        super().__init__()
        self.n_bins=5
        self.input_size = 700//self.n_bins
        self.output_size = 20
        self.hidden_list = hidden_list
        self.hidden_size = 128
        self.kernel_size = kernel_size
        self.beta = beta
        self.device = device
        self.surrogate_grad_type = surrogate_grad_type
        self.active_consumption = 0.1
        self.idle_consumption = 0.1
        self.stride = None
        self.padding = None

    def make_snn(self):
        if self.surrogate_grad_type == "atan":
            # neuron and simulation parameters
            spike_grad = snn.surrogate.atan()
        else:
            spike_grad = snn.surrogate.sigmoid()

        net = nn.Sequential(
            nn.Linear(
                700,
                128,
            ),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
        )
        net.extend([
            nn.Linear(
                128,
                128,
            ),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
        ])
        net.extend([
            nn.Linear(
                128,
                20,
            ),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
        ])
        net.to(self.device)

        return net


def forward_pass(net, data, active_consumption=None, idle_consumption=None):
    spk_rec = []
    mem_rec = []
    total_consumption = 0
    snn.utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        logger.info(data.size())
        logger.info(data[step].size())
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
