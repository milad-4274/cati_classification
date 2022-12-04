from ray import tune
from ray.tune.schedulers import ASHAScheduler
from training import PyTorchTrainable
import ray
from data import load_data

import os

config = {
    "hidden_units": tune.grid_search([ 64, 128]),
    "drop_rate": tune.uniform(0.0, 0.8),
    # "activation": tune.choice([nn.ReLU(True), nn.ELU(True), nn.SELU(True)]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.1, 0.9),
    "data" : load_data(os.path.join(os.path.abspath(os.getcwd()),"MYCATI/"),64),
    "base_model": tune.choice(["vgg"]),
    "use_classifier": tune.choice([True]),
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
    num_samples=2, # runs 15 jobs with separate sample from the search space
    checkpoint_at_end=True,
    checkpoint_freq=3,    
    scheduler=scheduler,
    stop={"training_iteration": 5},
    resources_per_trial={"cpu": 20, "gpu": 1}
   

   
)
stop = time()