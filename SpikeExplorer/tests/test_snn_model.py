import pytest

from torch import nn
from snn_model import SNN


@pytest.fixture
def config():
    return {
        "name": "experiment_name",
        "overwrite_existing_experiment": True,
        "is_test": False,
        "num_trials": 10,
        "batch_size": 32,
        "input_size": 2,
        "hidden_list": [12, 24, 12],
        "output_size": 32,
        "kernel_size": 5,
        "beta": 0.95,
        "surrogate_grad_type": "atan",
        "num_epochs": 10,
        "num_steps": 10,
        "objectives": [
            {
                "obj_name": "accuracy",
                "minimize": False,
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


def test_make_snn(config):
    model = SNN(
        input_size=config.get("input_size"),
        output_size=config.get("output_size"),
        hidden_list=config.get("hidden_list"),
        kernel_size= config.get("kernel_size"),
        beta=config.get("beta"),
        device=config.get("device"),
        surrogate_grad_type=config.get("surrogate_grad_type"),
    ).make_snn()
    assert isinstance(model, nn.Module)


def test_forward():
    pass
