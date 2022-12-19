import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

def parse_folder_name(folder_name):
    splitted = folder_name.split(",")
    base_model = splitted[0][-17:]
    lr = splitted[1]
    loss = splitted[2]
    momentum = splitted[3].split("_")[0]
    info = [base_model, lr,loss, momentum]
    print(info)
    return {spec.split('=')[0] : spec.split('=')[1] for spec in info}


folder_of_trainable_results = "hyper_tune_res"

result_folders = [file for file in os.listdir(folder_of_trainable_results) if os.path.isdir(os.path.join(folder_of_trainable_results,file))]
print(len(result_folders))

for folder in result_folders:
    json_path = os.path.join(folder_of_trainable_results,folder, "result.json")
    print(json_path)
    specs = parse_folder_name(folder)
    title_text = f"Model:{specs['base_model']}, lr: {specs['learning_rate']}, Loss function: {specs['loss']}, momentum: {specs['momentum']}"
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
        fig.suptitle(title_text)
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
        
        plt.savefig(folder+".png")

