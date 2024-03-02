import json
import os
import matplotlib.pyplot as plt

# https://web.archive.org/web/20140419054040/http://oco-carbon.com:80/2012/07/31/find-pareto-frontiers-in-python/
def pareto_frontier(metric_x, metric_y, max_x=True, max_y=True, params_list=None):
    if params_list:
        sorted_metric_list = sorted([[metric_x[i], metric_y[i], params_list[i]] for i in range(len(metric_x))], reverse=max_x)
    else:
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
    pareto_front_params = [pair_params[2] for pair_params in pareto_front]
    return pareto_front_x, pareto_front_y, pareto_front_params


def plot_pareto_given_x_and_y(metrics_dict, x, y, max_x=True, max_y=True, params_list=None):
    p_frontX, p_frontY, p_front_params = pareto_frontier(metrics_dict[x], metrics_dict[y],
                                         max_x=max_x,
                                         max_y=max_y,
                                         params_list=params_list)
    plt.title(f"Pareto Front {x}/{y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.scatter(metrics_dict[x], metrics_dict[y])
    plt.plot(p_frontX, p_frontY)
    plt.savefig(f"../results/test_pareto_front_{x}_{y}.png")
    plt.clf()


if __name__ == "__main__":
    for filename in os.listdir('../results/'):
        if "_results.json" in filename:
            with open(os.path.join('../results/', filename), 'r') as config:
                data = json.load(config)
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
                    params_list.append(
                        {
                            "neuron_type": data.get(index).get("neuron_type", "lif"),
                            "hidden_layer_num": data.get(index).get("hidden_layer_num", None),
                            "num_steps": data.get(index).get("num_steps", None),
                        }
                    )
                if "accuracy" in metrics_dict.keys() and "area" in metrics_dict.keys():
                    plot_pareto_given_x_and_y(metrics_dict, "accuracy", "area", max_x=True, max_y=False,
                                              params_list=params_list.copy())
                if "accuracy" in metrics_dict.keys() and "time" in metrics_dict.keys():
                    plot_pareto_given_x_and_y(metrics_dict, "accuracy", "time", max_x=True, max_y=False, params_list=params_list.copy())

                if "accuracy" in metrics_dict.keys() and "consumption" in metrics_dict.keys():
                    plot_pareto_given_x_and_y(metrics_dict, "accuracy", "consumption", max_x=True, max_y=False, params_list=params_list.copy())
                if "area" in metrics_dict.keys() and "time" in metrics_dict.keys():
                    plot_pareto_given_x_and_y(metrics_dict, "area", "time", max_x=False, max_y=False, params_list=params_list.copy())
                if "area" in metrics_dict.keys() and "consumption" in metrics_dict.keys():
                    plot_pareto_given_x_and_y(metrics_dict, "area", "consumption", max_x=False, max_y=False, params_list=params_list.copy())
                if "consumption" in metrics_dict.keys() and "time" in metrics_dict.keys():
                    plot_pareto_given_x_and_y(metrics_dict, "consumption", "time", max_x=False, max_y=False, params_list=params_list.copy())