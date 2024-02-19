import time

import numpy as np
import torch
from logging import Logger
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.logger import get_logger
from torch import nn, optim
from utils import load_nmnist, load_shd, load_mnist, load_dvs
from models import model_generator

logger: Logger = get_logger(__name__)

torch.manual_seed(104)


class AxManager:
    def __init__(self, config: dict):
        self.experiment_name = config.get("name")

        self.dtype = torch.float
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.overwrite_existing_experiment = config.get("overwrite_existing_experiment")
        self.is_test = config.get("is_test")
        self.num_trials = config.get("num_trials")

        self.objectives_dict: dict = {}
        self.parameters_list: list = []

        self.batch_size: int = config.get("batch_size")
        self.test_loader = None
        self.train_loader = None

        self.num_epochs: int = config.get("num_epochs")
        self.num_steps: int = config.get("num_steps")

        self.load_objectives(config.get("objectives"))
        self.load_parameters_to_search(config.get("parameters_to_search"))

        self.dataset = config.get("dataset")
        if self.dataset == "nmnist":
            self.train_loader, self.test_loader = load_nmnist(self.batch_size)
        elif self.dataset == "shd":
            self.hidden_layers = config.get("hidden_layers")
            self.output_size = 20
            self.train_loader, self.test_loader, self.input_size = load_shd(self.batch_size)
        elif self.dataset == "dvs":
            self.hidden_layers = config.get("hidden_layers")
            self.output_size = 11
            self.train_loader, self.test_loader, self.input_size = load_dvs(self.batch_size)
        else:
            self.input_size = 784
            self.hidden_layers = config.get("hidden_layers")
            self.output_size = 10
            self.train_loader, self.test_loader = load_mnist(self.batch_size)

        self.idle_consumption = config.get("neuron_consumption").get("idle")  # TODO: TO BE DEFINED FINAL METHOD
        self.active_consumption = config.get("neuron_consumption").get("active")  # TODO: TO BE DEFINED FINAL METHOD

    def load_parameters_to_search(self, parameters_to_search=None):
        if parameters_to_search:
            for param in parameters_to_search:
                self.parameters_list.append(param)
        else:
            self.parameters_list = [
                {
                    "name": "lr",
                    "type": "range",
                    "bounds": [1e-6, 0.4],
                    "log_scale": True,
                    "value_type": "float",
                },
                {
                    "name": "betas",
                    "type": "range",
                    "bounds": [0.5, 0.999],
                    "log_scale": False,
                    "value_type": "float",
                },
            ]

    def load_objectives(self, objectives):
        if objectives:
            for obj in objectives:
                self.objectives_dict[obj.get("obj_name")] = ObjectiveProperties(
                    minimize=obj.get("minimize"), threshold=obj.get("threshold")
                )
        else:
            self.objectives_dict = {"accuracy": ObjectiveProperties(minimize=False)}

    def run_experiment(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            name=self.experiment_name,
            parameters=self.parameters_list,
            objectives=self.objectives_dict,
            overwrite_existing_experiment=self.overwrite_existing_experiment,
            is_test=self.is_test,
        )

        for i in range(self.num_trials):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=self.train_evaluate(parameters)
            )

        logger.info(ax_client.experiment.optimization_config)

    def train_evaluate(self, parameterization):
        if self.dataset == "nmnist":
            pass
        elif self.dataset == "shd":
            model = model_generator.Net(
                input_size=self.input_size,
                hidden_layers=self.hidden_layers,
                output_size=self.output_size,
                beta=parameterization.get("exp_decay", 0.95),
                time_steps=parameterization.get("time_steps", self.num_steps),
                dataset=self.dataset
            )
            loss = nn.NLLLoss()
            log_softmax_fn = nn.LogSoftmax(dim=1)
        elif self.dataset == "dvs":
            model = model_generator.Net(
                input_size=self.input_size,
                hidden_layers=self.hidden_layers,
                output_size=self.output_size,
                beta=parameterization.get("exp_decay", 0.95),
                time_steps=parameterization.get("time_steps", self.num_steps),
                dataset=self.dataset
            )
            loss = nn.NLLLoss()
            log_softmax_fn = nn.LogSoftmax(dim=1)
        else:
            model = model_generator.Net(
                input_size=self.input_size,
                hidden_layers=self.hidden_layers,
                output_size=self.output_size,
                beta=parameterization.get("exp_decay", 0.95),
                time_steps=parameterization.get("time_steps", self.num_steps)
            )
            loss = nn.CrossEntropyLoss()

        print(model)

        optimizer = optim.Adam(
            model.parameters(),
            lr=parameterization.get("lr", 0.001),
            betas=(parameterization.get("betas", 0.9), 0.999),
        )

        loss_hist = []
        test_loss_hist = []
        accuracy = 0
        counter = 0
        eval_time = []
        cuda_eval_time = []
        consumptions = []

        # Outer training loop
        for epoch in range(self.num_epochs):
            train_batch = iter(self.train_loader)

            for data, targets in train_batch:
                data = data.to(self.device)
                targets = targets.to(self.device)

                model.train()
                if self.dataset == "shd":
                    spk_rec, mem_rec = model(data[:, :, 0, :])
                    m, _ = torch.max(mem_rec, 0)
                    log_p_y = log_softmax_fn(m)
                    loss_val = loss(log_p_y, targets)
                elif self.dataset == "dvs":
                    spk_rec, mem_rec = model(
                        data[:, :, 0, :, :].reshape(
                            parameterization.get("time_steps", self.num_steps),
                            self.batch_size,
                            16384
                        )
                    )  # in questo frangente ho (num_step, num_batch, c, h, w) io voglio (num_step, num_batch, 0 fissato e h*W reshape)
                    m, _ = torch.max(mem_rec, 0)
                    log_p_y = log_softmax_fn(m)
                    loss_val = loss(log_p_y, targets)
                else:
                    spk_rec, mem_rec = model(data.view(self.batch_size, -1))

                    # initialize the loss & sum over time
                    loss_val = torch.zeros((1), dtype=self.dtype, device=self.device)
                    loss_val += loss(spk_rec.sum(0), targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                if counter % 50 == 0:
                    print(f"Iteration: {counter} \t Train Loss: {loss_val.item()}")
                counter += 1

                if counter == 100:
                    break

            # Test set
            with torch.no_grad():
                model.eval()
                test_data, test_targets = next(iter(self.test_loader))
                test_data = test_data.to(self.device)
                test_targets = test_targets.to(self.device)

                if torch.cuda.is_available():
                    starter, ender = torch.cuda.Event(
                        enable_timing=True
                    ), torch.cuda.Event(enable_timing=True)
                    # GPU-WARM-UP
                    for _ in range(5):
                        _ = model(test_data.flatten(1))
                    starter.record()

                else:
                    starter = time.time()
                # Test set forward pass
                if self.dataset == "shd":
                    test_spk, test_mem = model(test_data[:, :, 0, :])
                elif self.dataset == "dvs":
                    test_spk, test_mem = model(
                        test_data[:, :, 0, :, :].reshape(
                            parameterization.get("time_steps", self.num_steps),
                            self.batch_size,
                            16384
                        )
                    )
                else:
                    test_spk, test_mem = model(test_data.view(self.batch_size, -1))
                if torch.cuda.is_available():
                    ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    cuda_eval_time.append(curr_time)
                else:
                    ender = time.time()
                    elapsed_time = ender - starter
                    eval_time.append(elapsed_time)

                _, idx = test_spk.sum(dim=0).max(1)

                actual_acc = np.mean((test_targets == idx).detach().cpu().numpy())

                if actual_acc > accuracy:
                    accuracy = actual_acc

                # Test set loss
                test_loss = torch.zeros((1), dtype=self.dtype, device=self.device)
                for step in range(self.num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

        if torch.cuda.is_available():
            avg_time = np.mean(cuda_eval_time)
        else:
            avg_time = np.mean(eval_time)

        resulting_dict_for_ax = {}
        for objective in self.objectives_dict.keys():
            if objective == "accuracy":
                resulting_dict_for_ax[objective] = accuracy
            elif objective == "time":
                resulting_dict_for_ax[objective] = avg_time
            elif objective == "consumption":
                resulting_dict_for_ax[objective] = torch.mean(torch.FloatTensor(consumptions))

        return resulting_dict_for_ax
