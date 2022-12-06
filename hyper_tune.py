from ray import tune
from ray.tune.schedulers import ASHAScheduler
from training import PyTorchTrainable
import ray
import torch.nn as nn
from data import load_data
import argparse
import os
import torch

gpu_n = 0
cpu_n = 0


parser = argparse.ArgumentParser(
                    prog = 'Hyperparameter Tuning',
                    description = 'check different hyperparameters using ray tune and pytorch',
                    epilog = 'you can change only number of cpus and gpus using command line.')


parser.add_argument('cpu', type=int, description="number of cpus to use for processing", default=2)           # positional argument
parser.add_argument('gpu', type=int, description="number of gpus to use for processing", default=1)           # positional argument


args = parser.parse_args()
if args.gpu > 0:
    if not torch.cuda.is_availabel():
        print("no gpu provided, continue using cpus")
        gpu_n = 0
cpu_n = args.cpu


config = {
    "hidden_units": tune.grid_search([ 64, 128, 256]),
    "drop_rate": tune.uniform(0.0, 0.8),
    "activation": tune.choice([nn.ReLU(True), nn.ELU(True), nn.SELU(True)]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.1, 0.9),
    "data" : load_data(os.path.join(os.path.abspath(os.getcwd()),"MYCATI/"),64),
    "base_model": tune.choice(["vgg","resnet"]),
    "use_classifier": tune.choice([True,False]),
    "loss": tune.choice(["focal","cross"]),

}

# Terminate less promising trials using early stopping
scheduler = ASHAScheduler(metric="mean_accuracy", mode="max")



from datetime import datetime
from time import time
ray.shutdown()

# ray.init(dashboard_host="0.0.0.0")

start = time()
# run trials
analysis = tune.run(
    PyTorchTrainable,
    config=config,
    num_samples=15, # runs 15 jobs with separate sample from the search space
    checkpoint_at_end=True,
    checkpoint_freq=3,    
    scheduler=scheduler,
    stop={"training_iteration": 50},
    resources_per_trial={"cpu": cpu_n, "gpu": gpu_n}
   

   
)
stop = time()