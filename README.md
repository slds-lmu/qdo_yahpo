In this repository we release all code to replicate all results, tables and figures presented in the paper:
A Collection of Quality Diversity Optimization Problems Derived from Hyperparameter Optimization of Machine Learning Models

The repository is structured as follows:
  * `pyribs_ranger.py` and `pyribs_xgboost.py` contain code for generating heatmaps on all benchmark problems; they can also be used as an entry point on how to set up YAHPO Gym for our QDO problems
  * `benchmark_ranger.py` and `benchmark_xgboost.py` are used for running the benchmark experiments
  * `random_emitter.py` contains code for a `RandomEmitter` to be used via [pyribs](https://pyribs.org/)
  * `helpers.py` contains helper functions
  * `Pipfile` and `requirements.txt` list all python module requirements
  * `Results/` contains benchmark results as `.csv` files
  * `Plots/` contains all plots as presented in the paper
  * `analysis.R` contains code to analyze benchmarks results and generate fancy ggplot plots

YAHPO Gym v1.0 was used.

Please see [here](https://github.com/slds-lmu/yahpo_gym) for more information about YAHPO Gym.
Detailed documentation on how to setup YAHPO Gym can be found [here](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym).
