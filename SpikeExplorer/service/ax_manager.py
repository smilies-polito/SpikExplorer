import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from logging import Logger
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.logger import get_logger
from torch import nn, optim

from service.utils import load_nmnist, load_shd, load_mnist, load_dvs
from models import model_generator

logger: Logger = get_logger(__name__)

torch.manual_seed(104)


class AxManager:
    def __init__(self, config: dict):
        self.experiment_name = config.get("name")
        self.results_path = config.get("results_path", "./results")
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

        self.neurons_occupation = config.get("neurons_occupation", {"lif": 1})
        self.weight_memory_occupation = config.get("weight_memory_occupation", 1)

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
                    "name": "beta1",
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
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        #   checkpoint_path = "./checkpoint"
        #   if not os.path.exists(checkpoint_path):
        #       os.makedirs(checkpoint_path)

        ax_client = AxClient()
        ax_client.create_experiment(
            name=self.experiment_name,
            parameters=self.parameters_list,
            objectives=self.objectives_dict,
            overwrite_existing_experiment=self.overwrite_existing_experiment,
            is_test=self.is_test,
        )

        #   TODO: future implementation
        #   if os.path.exists(f"{checkpoint_path}/checkpoint.json"):
        #       ax_client.load_from_json_file(f"{checkpoint_path}/checkpoint.json")

        for i in range(self.num_trials):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=self.train_evaluate(parameters)
            )

            #   ax_client.save_to_json_file(f"{checkpoint_path}/checkpoint.json")
            #   ax_client.save_to_json_file(f"{self.results_path}/{self.experiment_name}_after_trial_{i}.json")


        # objectives = ax_client.experiment.optimization_config.objective.objectives
        # accuracy_time_frontier = ax_plot.pareto_utils.compute_posterior_pareto_frontier(
        #     experiment=ax_client.experiment,
        #     data=ax_client.experiment.fetch_data(),
        #     primary_objective=objectives[0].metric,
        #     secondary_objective=objectives[1].metric,
        #     absolute_metrics=["accuracy", "time"],
        #     num_points=self.num_trials,
        # )
        # accuracy_time_plot_html = ax_render.plot_config_to_html(ax_plot.pareto_frontier.plot_pareto_frontier(accuracy_time_frontier, CI_level=0.90))
        # with open(f"{self.results_path}/{self.experiment_name}_pareto_accuracy_time.html", "w") as pareto_html:
        #     pareto_html.write(accuracy_time_plot_html)

        ax_client.save_to_json_file(f"{self.results_path}/{self.experiment_name}_completed.json")

        results = ax_client.get_trials_data_frame()
        metrics_dict = {}
        if "accuracy" in ax_client.objective_names:
            metrics_dict.update({"accuracy": results.get("accuracy").values})
        if "area" in ax_client.objective_names:
            metrics_dict.update({"area": results.get("area").values})
        if "time" in ax_client.objective_names:
            metrics_dict.update({"time": results.get("time").values})
        if "consumption" in ax_client.objective_names:
            metrics_dict.update({"consumption": results.get("consumption").values})
        if "accuracy" in ax_client.objective_names and "time" in ax_client.objective_names:
            self.plot_pareto_given_x_and_y(metrics_dict, "accuracy", "time", max_x=True, max_y=False)
        if "accuracy" in ax_client.objective_names and "area" in ax_client.objective_names:
            self.plot_pareto_given_x_and_y(metrics_dict, "accuracy", "area", max_x=True, max_y=False)
        if "accuracy" in ax_client.objective_names and "consumption" in ax_client.objective_names:
            self.plot_pareto_given_x_and_y(metrics_dict, "accuracy", "consumption", max_x=True, max_y=False)
        if "area" in ax_client.objective_names and "time" in ax_client.objective_names:
            self.plot_pareto_given_x_and_y(metrics_dict, "area", "time", max_x=False, max_y=False)
        if "area" in ax_client.objective_names and "consumption" in ax_client.objective_names:
            self.plot_pareto_given_x_and_y(metrics_dict, "area", "consumption", max_x=False, max_y=False)
        if "consumption" in ax_client.objective_names and "time" in ax_client.objective_names:
            self.plot_pareto_given_x_and_y(metrics_dict, "consumption", "time", max_x=False, max_y=False)
        data = results.to_dict()
        print(data)
        reformatted_dict = {}
        for trial in range(self.num_trials):
            reformatted_dict[trial] = {}
            for key in data.keys():
                reformatted_dict[trial][key] = data[key][trial]

        with open(f"{self.results_path}/{self.experiment_name}_results.json", "w") as json_file:
            json.dump(reformatted_dict, json_file)

        logger.info(ax_client.experiment.optimization_config)

    def plot_pareto_given_x_and_y(self, metrics_dict, x, y, max_x=True, max_y=True):
        p_frontX, p_frontY = self.pareto_frontier(metrics_dict[x], metrics_dict[y],
                                                       max_x=max_x,
                                                       max_y=max_y)
        plt.title(f"Pareto Front {x}/{y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.scatter(metrics_dict[x], metrics_dict[y])
        plt.plot(p_frontX, p_frontY)
        plt.savefig(f"{self.results_path}/{self.experiment_name}_pareto_front_{x}_{y}.png")
        plt.clf()

    def pareto_frontier(self, metric_x, metric_y, max_x=True, max_y=True):
        sorted_metric_list = sorted([[metric_x[i], metric_y[i]] for i in range(len(metric_x))], reverse=max_x)
        pareto_front = [sorted_metric_list[0]]
        for pair in sorted_metric_list[1:]:
            if max_y:
                if pair[1] >= pareto_front[-1][1]:
                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)
        pareto_front_x = [pair[0] for pair in pareto_front]
        pareto_front_y = [pair[1] for pair in pareto_front]
        return pareto_front_x, pareto_front_y

    def train_evaluate(self, parameterization):
        tau_mem = 10e-3
        tau_syn = 5e-3

        alpha = float(np.exp(-self.num_steps / tau_syn))
        beta = float(np.exp(-self.num_steps / tau_mem))

        hidden_layer_nums_param = parameterization.get("hidden_layer_num", None)
        hidden_layers_list = []

        if hidden_layer_nums_param:
            for i in range(hidden_layer_nums_param):
                hidden_layers_list.append(parameterization.get(f"hidden_layer_size_{i+1}"))
            self.hidden_layers = hidden_layers_list

        if self.dataset == "nmnist":
            model = None
        elif self.dataset == "shd":
            if parameterization.get("time_steps"):
                self.train_loader, self.test_loader, self.input_size = load_shd(
                    self.batch_size, parameterization.get("time_steps", self.num_steps)
                )
            model = model_generator.Net(
                input_size=self.input_size,
                hidden_layers=self.hidden_layers,
                output_size=self.output_size,
                #
                #    beta=parameterization.get("exp_decay", 0.95) if parameterization.get("learnable_exp_decay",
                #                                                                     False) else 0.95,
                #    {
                #        "name": "learnable_exp_decay",
                #        "type": "choice",
                #        "value_type": "bool",
                #        "values": [true, false]
                #    }
                alpha=parameterization.get("alpha", None) if parameterization.get("alpha", None) else alpha,
                beta=parameterization.get("beta", None) if parameterization.get("beta", None) else beta,
                learnable_exp_decay=parameterization.get("learnable_exp_decay", False),
                time_steps=parameterization.get("time_steps", self.num_steps),
                dataset=self.dataset,
                neuron_type=parameterization.get("neuron_type", "lif"),
            )
            loss = nn.CrossEntropyLoss()
            log_softmax_fn = nn.LogSoftmax(dim=1)
        elif self.dataset == "dvs":
            if parameterization.get("time_steps"):
                self.train_loader, self.test_loader, self.input_size = load_dvs(
                    self.batch_size, parameterization.get("time_steps", self.num_steps)
                )
            model = model_generator.Net(
                input_size=self.input_size,
                hidden_layers=self.hidden_layers,
                output_size=self.output_size,
                alpha=parameterization.get("alpha", None) if parameterization.get("alpha", None) else alpha,
                beta=parameterization.get("beta", None) if parameterization.get("beta", None) else beta,
                learnable_exp_decay=parameterization.get("learnable_exp_decay", False),
                time_steps=parameterization.get("time_steps", self.num_steps),
                dataset=self.dataset,
                neuron_type=parameterization.get("neuron_type", "lif"),
            )
            loss = nn.CrossEntropyLoss()
            log_softmax_fn = nn.LogSoftmax(dim=1)
        else:
            model = model_generator.Net(
                input_size=self.input_size,
                hidden_layers=self.hidden_layers,
                output_size=self.output_size,
                alpha=parameterization.get("alpha", None) if parameterization.get("alpha", None) else alpha,
                beta=parameterization.get("beta", None) if parameterization.get("beta", None) else beta,
                learnable_exp_decay=parameterization.get("learnable_exp_decay", False),
                time_steps=parameterization.get("time_steps", self.num_steps),
                neuron_type=parameterization.get("neuron_type", "lif"),
            )
            loss = nn.CrossEntropyLoss()

        print(model)
        num_neurons = 0
        num_weights = self.input_size * hidden_layers_list[0]
        for i in range(len(hidden_layers_list)):
            if i != (len(hidden_layers_list)-1):
                num_neurons += hidden_layers_list[i]
                num_weights += hidden_layers_list[i] * hidden_layers_list[i+1]
            else:
                num_neurons += hidden_layers_list[i] + self.output_size
                num_weights += hidden_layers_list[i] * self.output_size

        total_area = self.weight_memory_occupation * num_weights + self.neuron_memory_occupation_by_type(parameterization.get("neuron_type", "lif"))

        optimizer = optim.Adam(
            model.parameters(),
            lr=parameterization.get("lr", 0.001),
            betas=(parameterization.get("beta1", 0.9), 0.999),
        )

        loss_hist = []
        test_loss_hist = []
        accuracy = 0
        counter = 0
        eval_time = []
        cuda_eval_time = []

        # Outer training loop
        for epoch in range(self.num_epochs):
            train_batch = iter(self.train_loader)

            for data, targets in train_batch:
                data = data.to(self.device)
                targets = targets.to(self.device)

                model.train()
                if self.dataset == "shd":
                    spk_rec, mem_rec, total_consumption = model(data[:, :, 0, :])
                    loss_val = torch.zeros((1), dtype=self.dtype, device=self.device)
                    loss_val += loss(spk_rec.sum(0), targets)
                elif self.dataset == "dvs":
                    spk_rec, mem_rec, total_consumption = model(
                        data[:, :, 0, :, :].reshape(
                            parameterization.get("time_steps", self.num_steps),
                            self.batch_size,
                            16384
                        )
                    )
                    loss_val = torch.zeros((1), dtype=self.dtype, device=self.device)
                    loss_val += loss(spk_rec.sum(0), targets)
                else:
                    spk_rec, mem_rec, total_consumption = model(data.view(self.batch_size, -1))

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
                    starter.record()

                else:
                    starter = time.time()
                # Test set forward pass
                if self.dataset == "shd":
                    test_spk, test_mem, test_total_consumption = model(test_data[:, :, 0, :])
                elif self.dataset == "dvs":
                    test_spk, test_mem, test_total_consumption = model(
                        test_data[:, :, 0, :, :].reshape(
                            parameterization.get("time_steps", self.num_steps),
                            self.batch_size,
                            16384
                        )
                    )
                else:
                    test_spk, test_mem, test_total_consumption = model(test_data.view(self.batch_size, -1))
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
                resulting_dict_for_ax[objective] = test_total_consumption
            elif objective == "area":
                resulting_dict_for_ax[objective] = total_area

        return resulting_dict_for_ax

    def neuron_memory_occupation_by_type(self, neuron_type):
        return self.neurons_occupation.get(neuron_type, 1)


