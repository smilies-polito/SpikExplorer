import json
import sys

import service.ax_manager

default_config = {
    "dataset":"nmnist",
    "name": "experiment_name",
    "overwrite_existing_experiment": True,
    "is_test": False,
    "num_trials": 10,
    "batch_size": 32,
    "input_size": 2,
    "hidden_list": [12, 32],
    "output_size": 10,
    "kernel_size": 5,
    "beta": 0.95,
    "surrogate_grad_type": "atan",
    "num_epochs": 1,
    "num_steps": 5,
    "objectives": [
        {
            "obj_name": "accuracy",
            "minimize": False,
            "threshold": 0.9
        },
        {
            "obj_name": "time",
            "minimize": True,
            "threshold": 60
        },
        {
            "obj_name": "consumption",
            "minimize": True,
            "threshold": None
        }
    ],
    "parameters_to_search": [
        {
            "name": "lr",
            "type": "range",
            "bounds": [1e-6, 0.4],
            "log_scale": True,
            "value_type": "float"
        },
        {
            "name": "beta1",
            "type": "range",
            "bounds": [0.5, 0.999],
            "log_scale": False,
            "value_type": "float"
        }
    ],
    "neuron_consumption": {
        "active": 0.1,
        "idle": 0.001
    }
}


def main():
    ax_manager = service.ax_manager.AxManager(default_config)
    ax_manager.run_experiment()


if __name__ == "__main__":
    main()
