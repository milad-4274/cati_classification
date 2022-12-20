import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

def parse_folder_name(folder_name):
    splitted = folder_name.split(",")
    print(f"splitted: {splitted}")
    specs = {}
    for i,part in enumerate(splitted):
        part_splitted = part.split("=")
        key = part_splitted[0]
        val = part_splitted[1]
        
        if i == 0:
            key = '_'.join(part_splitted[0].split("_")[-2:])
        if i == len(splitted) -1 :
            val = part_splitted[1].split("_")[0]
        
        specs[key] = val
    return specs




folder_of_trainable_results = "PyTorchTrainable_2022-12-18_12-55-27"
destination_foder = "hyper_results/vgg"
if os.path.isdir(destination_foder):
    pass
else:
    os.mkdir(destination_foder)

result_folders = [file for file in os.listdir(folder_of_trainable_results) if os.path.isdir(os.path.join(folder_of_trainable_results,file))]
print(len(result_folders))

for folder in result_folders:
    json_path = os.path.join(folder_of_trainable_results,folder, "result.json")
    print(json_path)
    specs = parse_folder_name(folder)
    title_text = f"hyperparameter info: {specs}"
    with open(json_path, "r") as json_file:
        iteration_results = [json.loads(line) for line in json_file.readlines() ]
        print(type(iteration_results), len(iteration_results))
        train_loss, train_acc, test_loss, test_acc = [],[],[],[]
        for iter in iteration_results:
            train_acc.append(float(iter["tr_acc"])) 
            train_loss.append(float(iter["trloss"])) 
            test_loss.append(float(iter["tst_loss"])) 
            test_acc.append(float(iter["tst_acc"])) 

        fig,ax = plt.subplots(2,1,constrained_layout = True)
        fig.suptitle("\n".join(wrap(title_text,60)),size="small")
        print(title_text)
        ax[0].plot(train_acc,label="train")
        ax[0].plot(test_acc,label="test")
        ax[0].set_xlabel("iterations")
        ax[0].set_ylabel("accuracy")
        ax[0].set_title("accuracies of train and test")
        ax[0].legend()
        

        ax[1].plot(train_loss, label="train")
        ax[1].plot(test_loss, label="test")
        ax[1].set_xlabel("iterations")
        ax[1].set_ylabel("loss value")
        ax[1].set_title("loss values of train and test")
        ax[1].legend()
        
        plt.savefig(destination_foder + "/" +specs.__repr__()+".png")

