from ray import tune
from ray.tune.schedulers import ASHAScheduler
from training import PyTorchTrainable
from ray.tune import CLIReporter
# import ray
# import torch.nn as nn

import argparse
# import os
import torch
from ray.tune.search.hyperopt import HyperOptSearch
from ray import air
gpu_n = 0
cpu_n = 0


parser = argparse.ArgumentParser(
    prog='Hyperparameter Tuning',
    description='check different hyperparameters using ray tune and pytorch',
    epilog='you can change only number of cpus and gpus using command line.')


# positional argument
parser.add_argument('--cpu', type=int,
                    help="number of cpus to use for processing", default=2)
# positional argument
parser.add_argument('--gpu', type=int,
                    help="number of gpus to use for processing", default=1)


args = parser.parse_args()
if args.gpu > 0:
    if not torch.cuda.is_available():
        print("no gpu provided, continue using cpus")
        gpu_n = 0
    else:
        gpu_n = args.gpu
cpu_n = args.cpu

# gpu_n = 1
# cpu_n = 10

print(f"using {cpu_n} number of CPU core and {gpu_n} GPU(s)")

search_space = {
    "model": tune.choice(["vgg"]),
    "learning_rate": tune.loguniform(1e-3, 1e-1),
    "momentum": tune.uniform(0.1, 0.9),
    "base_model": tune.choice(["resnet"]),
    "loss": tune.choice(["cross", "focal"]),

}

# Terminate less promising trials using early stopping
scheduler = ASHAScheduler(metric="mean_accuracy", mode="max")


search_alg = HyperOptSearch()

if gpu_n > 0:
    trainable = tune.with_resources(
        PyTorchTrainable, {"gpu": gpu_n, "cpu": cpu_n})
else:
    trainable = PyTorchTrainable


reporter = CLIReporter(max_progress_rows=50)
metric_list = ["trloss", "tr_acc", "tst_acc", "tst_loss"]
for metric in metric_list:
    reporter.add_metric_column(metric)


tuner = tune.Tuner(
    trainable,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=10,
        metric="tst_loss",
        mode="min",
        max_concurrent_trials=10,
        search_alg=search_alg
    ),
    run_config=air.RunConfig(
        progress_reporter=reporter,
        stop=tune.stopper.MaximumIterationStopper(100)
        )
)


results = tuner.fit()
