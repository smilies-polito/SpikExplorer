import time
import datetime
import numpy as np
import torch
import tonic
from logging import Logger
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tonic import DiskCachedDataset
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.utils.common.logger import get_logger
from models.snn_model import SNN, forward_pass

torch.manual_seed(104)
logger: Logger = get_logger(__name__)

class AxManager:

    def __init__(self, config: dict):

        self.experiment_name = config.get("name")

        self.dtype = torch.float
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.overwrite_existing_experiment = config.get("overwrite_existing_experiment")
        self.is_test = config.get("is_test")
        self.num_trials = config.get("num_trials")

        self.objectives_dict: dict = {}
        self.parameters_list: list = []

        self.batch_size: int = config.get("batch_size")
        self.test_loader = None
        self.train_loader = None

        self.input_size: int = config.get("input_size")
        self.hidden_list: list = config.get("hidden_list")
        self.output_size: int = config.get("output_size")
        self.kernel_size: int = config.get("kernel_size")
        self.beta = config.get("beta")

        if config.get("surrogate_grad_type") is not None:
            self.surrogate_grad_type = config.get("surrogate_grad_type")
        else:
            self.surrogate_grad_type = "atan"

        self.num_epochs: int = config.get("num_epochs")
        self.num_steps: int = config.get("num_steps")

        self.load_objectives(config.get("objectives"))
        self.load_parameters_to_search(config.get("parameters_to_search"))

        if config.get("dataset") == "nmnist":
            self.load_nmnist()
        elif config.get("dataset") == "svd":
            self.load_svd()
        else:
            self.load_mnist()

        self.idle_consumption = config.get("neuron_consumption").get("idle")
        self.active_consumption = config.get("neuron_consumption").get("active")

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
                    "value_type": 'float'
                },
                {
                    "name": "beta1",
                    "type": "range",
                    "bounds": [0.5, 0.999],
                    "log_scale": False,
                    "value_type": 'float'
                }
            ]

    def load_objectives(self, objectives):
        if objectives:
            for obj in objectives:
                self.objectives_dict[obj.get("obj_name")] = ObjectiveProperties(
                    minimize=obj.get("minimize"),
                    threshold=obj.get("threshold")
                )
        else:
            self.objectives_dict = {
                "accuracy": ObjectiveProperties(minimize=False)
            }

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
            ax_client.complete_trial(trial_index=trial_index, raw_data=self.train_evaluate(parameters))
        objectives = ax_client.experiment.optimization_config.objective.objectives
        frontier = compute_posterior_pareto_frontier(
            experiment=ax_client.experiment,
            data=ax_client.experiment.fetch_data(),
            primary_objective=objectives[0].metric,
            secondary_objective=objectives[1].metric,
            absolute_metrics=["accuracy", "time"],
            num_points=10,
        )
        logger.info(f"frontier points found {frontier}")
        fig = plot_pareto_frontier(frontier)
        # create an Ax report
        with open('report.html', 'w') as outfile:
            outfile.write(
                render_report_elements(
                    f"frontier_{datetime.datetime.now()}",
                    html_elements=[plot_config_to_html(fig)],
                    header=False,
                )
            )

    def load_mnist(self):

        data_path = './data/mnist'

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        self.train_loader = DataLoader(mnist_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(mnist_test, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def load_nmnist(self):

        sensor_size = tonic.datasets.NMNIST.sensor_size

        # Denoise removes isolated, one-off events
        # time_window
        # ToFrame transformation, which reduces temporal precision but also allows us to work with it in a dense format
        frame_transform = transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(
                sensor_size=sensor_size,
                time_window=1000)
        ])

        trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
        testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

        transform = tonic.transforms.Compose([
            torch.from_numpy,
            transforms.RandomRotation([-10, 10])
        ])

        cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

        # no augmentations for the testset
        cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

        self.train_loader = DataLoader(cached_trainset, batch_size=self.batch_size,
                                       collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
        self.test_loader = DataLoader(cached_testset, batch_size=self.batch_size,
                                      collate_fn=tonic.collation.PadTensors(batch_first=False))

    def load_svd(self):
        pass

    def train_evaluate(self, parameterization):
        model = SNN(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_list=self.hidden_list,
            kernel_size=self.kernel_size,
            beta=self.beta,
            device=self.device,
            surrogate_grad_type=self.surrogate_grad_type
        ).make_snn()

        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=parameterization.get("lr", 0.001),
                               betas=(parameterization.get("beta1", 0.9), 0.999))

        """
        def print_batch_accuracy(data, targets, train=False):
            output, _ = model(data.view(self.batch_size, -1))
            _, idx = output.sum(dim=0).max(1)
            acc = np.mean((targets == idx).detach().cpu().numpy())

            if train:
                print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
            else:
                print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")
        """

        loss_hist = []
        test_loss_hist = []
        accuracy = 0
        counter = 0
        eval_time = []
        cuda_eval_time = []
        consumptions = []

        """
        # Correct way to measure elapsed time during evaluations
    
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        for _ in range(10):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(mean_syn)
        """

        # Outer training loop
        for epoch in range(self.num_epochs):
            iter_counter = 0

            for i, (data, targets) in enumerate(iter(self.train_loader)):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # forward pass
                model.train()
                spk_rec, mem_rec, total_consumption = forward_pass(model, data, self.active_consumption, self.idle_consumption)

                # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=self.dtype, device=self.device)
                for step in range(self.num_steps):
                    loss_val += loss(mem_rec[step], targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                # Test set
                with torch.no_grad():
                    model.eval()
                    test_data, test_targets = next(iter(self.test_loader))
                    test_data = test_data.to(self.device)
                    test_targets = test_targets.to(self.device)

                    if torch.cuda.is_available():
                        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        # GPU-WARM-UP
                        for _ in range(5):
                            _ = forward_pass(model, test_data)
                        starter.record()

                    else:
                        starter = time.time()
                    # Test set forward pass
                    test_spk, test_mem, total_consumption = forward_pass(
                        model,
                        test_data,
                        self.active_consumption,
                        self.idle_consumption
                    )
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

                    consumptions.append(total_consumption)

                    counter += 1
                    iter_counter += 1

        if torch.cuda.is_available():
            avg_time = np.mean(cuda_eval_time)
        else:
            avg_time = np.mean(eval_time)
        avg_consumption = np.mean(consumptions)
        return {"accuracy": accuracy, "time": avg_time, "consumption": avg_consumption}
