from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.iaml
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer
from random_emitter import RandomEmitter
from helpers import trafo, retrafo
import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap
import numpy as np
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def evaluate_ranger(x, features, params, bench, task_id, lower, upper, trafo=True):
    if trafo:
        x = np.array([retrafo(i, lower, upper) for i in x])
    # YAHPO Gym supports batch predicts if configs are passed as a list of dicts
    config = [dict(zip(params, x[i])) for i in range(x.shape[0])]
    # add the other hyperparameters not part of the search space and respect integers
    for i in range(len(config)):
        config[i].update({"num.trees":round(config[i]["num.trees"]), "min.node.size":round(config[i]["min.node.size"])})
        config[i].update({"task_id":task_id, "trainsize":1, "replace":"TRUE", "respect.unordered.factors":"ignore", "splitrule":"gini"})
    results = bench.objective_function(config)
    targets = ["mmce"]
    targets.extend(features)
    y = pd.DataFrame(results)[targets]
    y[["mmce"]] = 1 - y[["mmce"]]  # pyribs maximizes by default so we turn mmce into acc
    return y.values  # pyribs expects a numpy array as return value

def run_ranger_interpretability(task_id):
    bench = benchmark_set.BenchmarkSet("iaml_ranger", check = False)
    bench.set_instance(task_id)
    search_space = bench.get_opt_space()

    # num.trees, mtry.ratio, min.node.size, sample.fraction
    params = ["num.trees", "mtry.ratio", "min.node.size", "sample.fraction"]
    bounds = [(search_space.get_hyperparameter(param).lower, search_space.get_hyperparameter(param).upper) for param in params]
    defaults = [100., 0.1, 1., 1.]
    lower = [bound[0] for bound in bounds]
    upper = [bound[1] for bound in bounds]

    normalized_bounds = [(0., 1.) for i in range(len(bounds))]
    normalized_defaults = trafo(defaults, lower, upper)

    if task_id == "41146":
        archive = GridArchive([21, 100], [(0., 20.), (0., 1.)])
    elif task_id == "40981":
        archive = GridArchive([15, 100], [(0., 14.), (0., 1.)])
    elif task_id == "1489":
        archive = GridArchive([6, 100], [(0., 5.), (0., 1.)])
    elif task_id == "1067":
        archive = GridArchive([22, 100], [(0., 21.), (0., 1.)])

    emitters = [
        RandomEmitter(
            archive=archive,
            x0=normalized_defaults,
            bounds=normalized_bounds,
            batch_size=25,
            seed=1
        ),
        GaussianEmitter(
            archive=archive,
            x0=normalized_defaults,
            sigma0=0.05,
            bounds=normalized_bounds,
            batch_size=25,
            seed=2
        ),
        GaussianEmitter(
            archive=archive,
            x0=normalized_defaults,
            sigma0=0.1,
            bounds=normalized_bounds,
            batch_size=25,
            seed=3
        ),
        GaussianEmitter(
            archive=archive,
            x0=normalized_defaults,
            sigma0=0.2,
            bounds=normalized_bounds,
            batch_size=25,
            seed=4
        )
    ]

    optimizer = Optimizer(archive, emitters)

    total_itrs = 10000  # 1e+06 evaluations in total
    for itr in range(1, total_itrs + 1):
        x = optimizer.ask()
        results = evaluate_ranger(x, ["nf", "ias"], params, bench, task_id, lower, upper)
        optimizer.tell(results[:, 0], results[:, [1, 2]])
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive)
    plt.title("iaml_ranger " + task_id)
    plt.xlabel("NF")
    plt.ylabel("IAS")
    plt.savefig("Plots/iaml_ranger_ias_nf_" + task_id + ".pdf", dpi = 150)
    return None

def run_ranger_hardware(task_id):
    bench = benchmark_set.BenchmarkSet("iaml_ranger", check = False)
    bench.set_instance(task_id)
    search_space = bench.get_opt_space()

    # num.trees, mtry.ratio, min.node.size, sample.fraction
    params = ["num.trees", "mtry.ratio", "min.node.size", "sample.fraction"]
    bounds = [(search_space.get_hyperparameter(param).lower, search_space.get_hyperparameter(param).upper) for param in params]
    defaults = [100., 0.1, 1., 1.]
    lower = [bound[0] for bound in bounds]
    upper = [bound[1] for bound in bounds]

    normalized_bounds = [(0., 1.) for i in range(len(bounds))]
    normalized_defaults = trafo(defaults, lower, upper)

    if task_id == "41146":
        archive = GridArchive([33, 33], [(1., 200.), (0.19, 4.5)])
    elif task_id == "40981":
        archive = GridArchive([33, 33], [(1., 40.), (0.10, 0.65)])
    elif task_id == "1489":
        archive = GridArchive([33, 33], [(1., 200.), (0.19, 4.5)])
    elif task_id == "1067":
        archive = GridArchive([33, 33], [(1., 78.), (0.13, 1.55)])

    emitters = [
        RandomEmitter(
            archive=archive,
            x0=normalized_defaults,
            bounds=normalized_bounds,
            batch_size=25,
            seed=1
        ),
        GaussianEmitter(
            archive=archive,
            x0=normalized_defaults,
            sigma0=0.05,
            bounds=normalized_bounds,
            batch_size=25,
            seed=2
        ),
        GaussianEmitter(
            archive=archive,
            x0=normalized_defaults,
            sigma0=0.1,
            bounds=normalized_bounds,
            batch_size=25,
            seed=3
        ),
        GaussianEmitter(
            archive=archive,
            x0=normalized_defaults,
            sigma0=0.2,
            bounds=normalized_bounds,
            batch_size=25,
            seed=4
        )
    ]

    optimizer = Optimizer(archive, emitters)

    total_itrs = 10000  # 1e+06 evaluations in total
    for itr in range(1, total_itrs + 1):
        x = optimizer.ask()
        results = evaluate_ranger(x, ["rammodel", "timepredict"], params, bench, task_id, lower, upper)
        optimizer.tell(results[:, 0], results[:, [1, 2]])
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive)
    plt.title("iaml_ranger " + task_id)
    plt.xlabel("rammodel")
    plt.ylabel("timepredict")
    plt.savefig("Plots/iaml_ranger_rammodel_timepredict_" + task_id + ".pdf", dpi = 150)
    return None

if __name__ == "__main__":
    run_ranger_interpretability("41146")
    run_ranger_interpretability("40981")
    run_ranger_interpretability("1489")
    run_ranger_interpretability("1067")
    run_ranger_hardware("41146")
    run_ranger_hardware("40981")
    run_ranger_hardware("1489")
    run_ranger_hardware("1067")

