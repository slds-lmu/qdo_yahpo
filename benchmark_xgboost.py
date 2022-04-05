from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.iaml
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter
from ribs.optimizers import Optimizer
from random_emitter import RandomEmitter
from helpers import trafo, retrafo
import numpy as np
import pandas as pd

def evaluate_xgboost(x, features, params, bench, task_id, lower, upper, trafo=True):
    if trafo:
        x = np.array([retrafo(i, lower, upper) for i in x])
    # YAHPO Gym supports batch predicts if configs are passed as a list of dicts
    config = [dict(zip(params, x[i])) for i in range(x.shape[0])]
    # add the other hyperparameters not part of the search space and respect log trafos and integers
    for i in range(len(config)):
        config[i].update({"alpha":np.exp(config[i]["alpha"]), "lambda":np.exp(config[i]["lambda"]), "eta":np.exp(config[i]["eta"]), "gamma":np.exp(config[i]["gamma"]), "min_child_weight":np.exp(config[i]["min_child_weight"])})
        config[i].update({"nrounds":round(np.exp(config[i]["nrounds"])), "max_depth":round(config[i]["max_depth"])})
        config[i].update({"task_id":task_id, "trainsize":1, "booster":"gbtree"})
    results = bench.objective_function(config)
    targets = ["mmce"]
    targets.extend(features)
    y = pd.DataFrame(results)[targets]
    y[["mmce"]] = 1 - y[["mmce"]]  # pyribs maximizes by default so we turn mmce into acc
    return y.values  # pyribs expects a numpy array as return value

def run_xgboost_interpretability(task_id, emitter_type, seed):
    bench = benchmark_set.BenchmarkSet("iaml_xgboost", check = False)  # we disable input checking of parameters for speed up
    bench.set_instance(task_id)
    search_space = bench.get_opt_space()

    # alpha, lambda, nrounds, subsample, colsample_bylevel, colsample_bytree, eta, gamma, max_depth, min_child_weight
    params = ["alpha", "lambda", "nrounds", "subsample", "colsample_bylevel", "colsample_bytree", "eta", "gamma", "max_depth", "min_child_weight"]
    bounds = [(search_space.get_hyperparameter(param).lower, search_space.get_hyperparameter(param).upper) for param in params]
    defaults = [0.001, 0.001, 1000., 1., 1., 1., 0.3, 0.001, 6., 3.]
    for i in range(len(params)):
        if search_space.get_hyperparameter(params[i]).log:
            bounds[i] = np.log(bounds[i])
            defaults[i] = np.log(defaults[i])
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

    if emitter_type == "random":
        emitters = [
            RandomEmitter(
                archive=archive,
                x0=normalized_defaults,
                bounds=normalized_bounds,
                batch_size=100,
                seed=seed
            )
        ]
    elif emitter_type == "gaussian":
        emitters = [
            GaussianEmitter(
                archive=archive,
                x0=normalized_defaults,
                sigma0=0.1,
                bounds=normalized_bounds,
                batch_size=100,
                seed=seed
            )
        ]
    elif emitter_type == "improvement":
        emitters = [
            ImprovementEmitter(
                archive=archive,
                x0=normalized_defaults,
                sigma0=0.1,
                selection_rule="filter",
                bounds=normalized_bounds,
                batch_size=100,
                seed=seed
            )
        ]
    elif emitter_type == "mixed":
        emitters = [
            GaussianEmitter(
                archive=archive,
                x0=normalized_defaults,
                sigma0=0.1,
                bounds=normalized_bounds,
                batch_size=50,
                seed=seed
            ),
            ImprovementEmitter(
                archive=archive,
                x0=normalized_defaults,
                sigma0=0.1,
                selection_rule="filter",
                bounds=normalized_bounds,
                batch_size=50,
                seed=seed+1
            )
        ]

    optimizer = Optimizer(archive, emitters)

    total_itrs = 1000  # 100000 evaluations in total
    stats = pd.DataFrame(columns=["num_elites", "coverage", "qd_score", "obj_max", "obj_mean"], index = range(total_itrs))
    for itr in range(1, total_itrs + 1):
        x = optimizer.ask()
        results = evaluate_xgboost(x, ["nf", "ias"], params, bench, task_id, lower, upper)
        optimizer.tell(results[:, 0], results[:, [1, 2]])
        stats.iloc[itr - 1] = archive.stats
    stats["iter"] = range(1, total_itrs + 1)
    return stats

if __name__ == "__main__":
    repls = 10
    for task_id in ["41146", "40981", "1489", "1067"]:
        df_task = [None] * repls
        for repl in range(repls):
            random = run_xgboost_interpretability(task_id, "random", repl)
            random["method"] = "random"
            gaussian = run_xgboost_interpretability(task_id, "gaussian", repl)
            gaussian["method"] = "gaussian"
            improvement = run_xgboost_interpretability(task_id, "improvement", repl)
            improvement["method"] = "improvement"
            mixed = run_xgboost_interpretability(task_id, "mixed", repl)
            mixed["method"] = "mixed"
            df_task[repl] = pd.concat([random, gaussian, improvement, mixed])
            df_task[repl]["repl"] = repl + 1
        res_task = pd.concat(df_task)
        res_task["task"] = task_id
        res_task.to_csv("Results/xgboost_interpretability_" + task_id + ".csv", index = False)

