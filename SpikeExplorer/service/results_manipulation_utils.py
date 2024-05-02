import json
import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from collections import defaultdict
import matplotlib.colorbar as colorbar
import matplotlib.colors as clr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# https://web.archive.org/web/20140419054040/http://oco-carbon.com:80/2012/07/31/find-pareto-frontiers-in-python/

dataset = "dvs"

def best_results_dataframe_from_dict_given_metric(document: dict, metric: str, maximize: bool = True, filter_param: str = None):
    """
    Args:
        document: dict file structured as
            {
                0: {
                    metric_a: value_metric_a,
                    metric_b: value_metric_b,
                    metric_c: value_metric_c,
                    ... ,
                    parameter_a: value_parameter_a,
                    parameter_b: value_parameter_b,
                    ...
                },
                ...
            }
        metric: str
        maximize: bool, if better high score for the metric set True else set False

    Returns: dict
    """
    best_results = {}

    for trial, trial_dict in document.items():
        filter_param_value = trial_dict[filter_param]
        metric_value = trial_dict[metric]
        if maximize:
            if filter_param_value not in best_results or metric_value > best_results[filter_param_value][metric]:
                best_results[filter_param_value] = trial_dict
        else:
            if filter_param_value not in best_results or metric_value < best_results[filter_param_value][metric]:
                best_results[filter_param_value] = trial_dict

    return best_results


def pareto_frontier(metric_x, metric_y, params_list, max_x=True, max_y=True):
    sorted_data = sorted(zip(metric_x, metric_y, params_list["neuron_type"], params_list["number_of_neuron_list"]), key=lambda d: d[0], reverse=max_x)
    pareto_front = [sorted_data[0]]
    for sorted_tuple in sorted_data:
        if max_y:
            if sorted_tuple[1] >= pareto_front[-1][1]:
                pareto_front.append(sorted_tuple)
        else:
            if sorted_tuple[1] <= pareto_front[-1][1]:
                pareto_front.append(sorted_tuple)
    return zip(*pareto_front)


def plot_pareto_given_x_y_and_params(metrics_dict, x_name, y_name, params_dict, max_x=True, max_y=True):
    pareto_front_x, pareto_front_y, pareto_front_neurons_type, pareto_front_neurons_num = pareto_frontier(
        metric_x=metrics_dict[x_name],
        metric_y=metrics_dict[y_name],
        params_list=params_dict,
        max_x=max_x,
        max_y=max_y
    )
    shape_idx = []
    label_color_dict = {}

    if params_dict.get("neuron_type", None):
        label_color_dict = {'lif': (0.6, 0, 0),
                            'rlif': (1, 0.5, 0),
                            'syn': (0, 0.4, 0),
                            'rsyn': (0, 0, 0.6)}
        all_labels = list(label_color_dict.keys())
        all_colors = list(label_color_dict.values())
        n_colors = len(all_colors)
        cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)

        color_idx = [all_colors.index(label_color_dict[label]) for label in metrics_dict["neuron_type"]]

    if params_dict.get("number_of_neuron_list", None):
        shape_list = []
        for num in params_dict.get("number_of_neuron_list", None):
            if num <= 150:
                shape_list.append("^")
            elif 250 >= num > 150:
                shape_list.append("s")
            elif num > 250:
                shape_list.append("o")

        shape_idx = shape_list

    plt.grid()

    if x_name == "area":
        plt.xlabel("Look Up Table (LUT) Equivalent",  fontsize=18)
    elif x_name == "consumption":
        plt.xlabel("Power (mW)",  fontsize=18)
    elif x_name == "time":
        plt.xlabel("Time",  fontsize=18)
    elif x_name == "accuracy":
        plt.xlabel("Accuracy",  fontsize=18)
    else:
        plt.xlabel(x_name)

    if y_name == "area":
        plt.ylabel("Look Up Table (LUT) Equivalent",  fontsize=18)
    elif y_name == "consumption":
        plt.ylabel("Power (mW)",  fontsize=18)
    elif y_name == "time":
        plt.ylabel("Time",  fontsize=18)
    elif y_name == "accuracy":
        plt.ylabel("Accuracy",  fontsize=18)
    else:
        plt.ylabel(y_name)

    if params_dict.get("number_of_neuron_list", None) and params_dict.get("neuron_type", None):
        for x, y, color, shape in zip(metrics_dict[x_name], metrics_dict[y_name], color_idx, shape_idx):
            plt.scatter(x, y, c=all_colors[color], cmap=cm, marker=shape)
    elif params_dict.get("number_of_neuron_list", None) and not params_dict.get("neuron_type", None):
        for x, y, shape in zip(metrics_dict[x_name], metrics_dict[y_name], shape_idx):
            plt.scatter(x, y, marker=shape)
    elif params_dict.get("neuron_type", None) and not params_dict.get("number_of_neuron_list", None):
        for x, y, color in zip(metrics_dict[x_name], metrics_dict[y_name], color_idx):
            plt.scatter(x, y, c=all_colors[color], cmap=cm)
    else:
        plt.scatter(metrics_dict[x_name], metrics_dict[y_name])

    plt.plot(pareto_front_x, pareto_front_y)
    if max_x and max_y:
        plt.hlines(pareto_front_y[0], min(metrics_dict[x_name]), pareto_front_x[0], linestyles='dashed')
        plt.vlines(pareto_front_x[-1], min(metrics_dict[y_name]), pareto_front_y[-1], linestyles='dashed')
    elif max_x and not max_y:
        plt.hlines(pareto_front_y[0], min(metrics_dict[x_name]), pareto_front_x[0], linestyles='dashed')
        plt.vlines(pareto_front_x[-1], pareto_front_y[-1], max(metrics_dict[y_name]), linestyles='dashed')
    elif not max_x and max_y:
        plt.hlines(pareto_front_y[-1], pareto_front_x[-1], max(metrics_dict[x_name]), linestyles='dashed')
        plt.vlines(pareto_front_x[0], min(metrics_dict[y_name]), pareto_front_y[0], linestyles='dashed')
    elif not max_x and not max_y:
        plt.hlines(pareto_front_y[-1], pareto_front_x[-1], max(metrics_dict[x_name]), linestyles='dashed')
        plt.vlines(pareto_front_x[0], pareto_front_y[0], max(metrics_dict[y_name]), linestyles='dashed')

    # triangle = mlines.Line2D([], [], marker='^', linestyle='None',
    #                           markersize=10, label='# Neurons <= 150')
    # square = mlines.Line2D([], [], marker='s', linestyle='None',
    #                            markersize=10, label='150 < # Neurons <= 250')
    # circle = mlines.Line2D([], [], marker='o', linestyle='None',
    #                                 markersize=10, label='# Neurons >= 250')

    square = mlines.Line2D([], [], marker='s', linestyle='None', markersize=10, label='# Neurons = 200')

    plt.legend(handles=[square])

    plt.autoscale()

    plt.savefig(f"../results/{dataset}_pareto_front_{x_name}_{y_name}.png")
    plt.clf()


def rescale_list(input_list: list, min_val, max_val):
    input_min, input_max = np.min(input_list), np.max(input_list)
    return [((i - input_min) * (max_val - min_val) / (input_max - input_min)) + min_val for i in input_list]


if __name__ == "__main__":
    for filename in os.listdir('../results/'):
        if "_results.json" in filename:
            with open(os.path.join('../results/', filename), 'r') as config:
                data = json.load(config)
                besties = best_results_dataframe_from_dict_given_metric(data, 'accuracy', filter_param='neuron_type')
                with open(f"../results/{dataset}_best_results.json", "w") as json_file:
                    json.dump(besties, json_file)
                metrics_dict = {}
                params_list = []
                for index in data.keys():
                    if "accuracy" in data.get(index).keys():
                        if "accuracy" not in metrics_dict.keys():
                            metrics_dict["accuracy"] = []
                        metrics_dict["accuracy"].append(data.get(index).get("accuracy"))
                    if "area" in data.get(index).keys():
                        if "area" not in metrics_dict.keys():
                            metrics_dict["area"] = []
                        metrics_dict["area"].append(data.get(index).get("area"))
                    if "time" in data.get(index).keys():
                        if "time" not in metrics_dict.keys():
                            metrics_dict["time"] = []
                        metrics_dict["time"].append(data.get(index).get("time"))
                    if "consumption" in data.get(index).keys():
                        if "consumption" not in metrics_dict.keys():
                            metrics_dict["consumption"] = []
                        metrics_dict["consumption"].append(data.get(index).get("consumption"))
                    if "neuron_type" in data.get(index).keys():
                        if "neuron_type" not in metrics_dict.keys():
                            metrics_dict["neuron_type"] = []
                        metrics_dict["neuron_type"].append(data.get(index).get("neuron_type"))
                    if "hidden_layer_num" in data.get(index).keys():
                        if "number_of_neuron_list" not in metrics_dict.keys():
                            metrics_dict["number_of_neuron_list"] = []
                        if data.get(index).get('hidden_layer_num') == 1:
                            metrics_dict["number_of_neuron_list"].append(
                                data.get(index).get("hidden_layer_size_1", 200))
                        elif data.get(index).get('hidden_layer_num') == 2:
                            metrics_dict["number_of_neuron_list"].append(
                                data.get(index).get("hidden_layer_size_1", 100) +
                                data.get(index).get("hidden_layer_size_2", 100))
                        elif data.get(index).get('hidden_layer_num') == 3:
                            metrics_dict["number_of_neuron_list"].append(
                                data.get(index).get("hidden_layer_size_1", 100) +
                                data.get(index).get("hidden_layer_size_2", 50) +
                                data.get(index).get("hidden_layer_size_3", 50)
                            )
                metrics_dict["consumption"] = rescale_list(metrics_dict["consumption"], 150, 850)
                metrics_dict["area"] = rescale_list(metrics_dict["area"], 30000, 360000)
                params_dict = {
                    "neuron_type": metrics_dict["neuron_type"],
                    "number_of_neuron_list": metrics_dict["number_of_neuron_list"]
                }

                plot_pareto_given_x_y_and_params(metrics_dict, "time", 'accuracy', params_dict, max_x=False, max_y=True)
                plot_pareto_given_x_y_and_params(metrics_dict, "area", 'time', params_dict, max_x=False, max_y=False)
                plot_pareto_given_x_y_and_params(metrics_dict, "time", 'consumption', params_dict, max_x=False, max_y=False)
                plot_pareto_given_x_y_and_params(metrics_dict, "area", 'consumption', params_dict, max_x=False, max_y=False)
                plot_pareto_given_x_y_and_params(metrics_dict, "area", 'accuracy', params_dict, max_x=False, max_y=True)
                plot_pareto_given_x_y_and_params(metrics_dict, "consumption", 'accuracy', params_dict, max_x=False, max_y=True)
