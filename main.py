from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# import seaborn as sns
import matplotlib.pyplot as plt

cudnn.benchmark = True
plt.ion()   # interactive mode


EPOCHS = 50
save_name = "res_1_ce_sgd_b64_train_val_ilr001"

from data import MyDataset, load_data
from training import train_model
from utils import plot_confusion_matrix, imshow, visualize_model, compute_confusion_matrix, plot_train_info
from losses import FocalLoss



batch_size = 32


data_dir = "MYCATI"

dataloaders, class_names, dataset_sizes = load_data(batch_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device is {device}")



# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

resnet18 = torchvision.models.resnet18(pretrained=True)
# nofreeze or freeze
for param in resnet18.parameters():
    param.requires_grad = False



# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 3)

resnet18 = resnet18.to(device)


# Parameters of newly constructed modules have requires_grad=True by default

# classifier = nn.Sequential(nn.Linear(num_ftrs, 512),
#                           nn.ReLU(),
#                           nn.Dropout(p=0.2),
#                           nn.Linear(512, 3),
#                            nn.LogSoftmax(dim=1))

# resnet18.fc = classifier
# resnet18 = resnet18.to(device)


# criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)

criterion = nn.CrossEntropyLoss().to(device)

# criterion = FocalLoss().to(device)

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(resnet18.fc.parameters(), lr=0.00220135, momentum=0.708253)
# optimizer_conv = optim.Adam(resnet18.fc.parameters(), lr=0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=20, gamma=1, verbose=True)


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
model_conv_resnet, resnet_train_info = train_model(resnet18, criterion, optimizer_conv,
                         exp_lr_scheduler,dataloaders,dataset_sizes,device, num_epochs=EPOCHS)


# print(train_info)
plot_train_info(resnet_train_info)
plt.savefig(f"res/{save_name}.png")


visualize_model(model_conv_resnet, dataloaders, class_names, device)
torch.save(model_conv_resnet.state_dict(), f"res/{save_name}.pt")


classes = ['DC', 'HY', 'WC']


correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

print("Resnet Reuslt on val data")

with torch.no_grad():
    for data in dataloaders['val']:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_conv_resnet(images)
        _, predictions = torch.max(outputs, 1)
        
        for label, prediction in zip(labels, predictions):
            if label == prediction:
              correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1



for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))


nb_classes = 3

print("confusion matrix for resnet")

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_conv_resnet(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
class_dict = {0:'DC', 1:'HY', 2:'WC'}

mat = compute_confusion_matrix(model=model_conv_resnet, data_loader=dataloaders['val'], device=torch.device('cuda'))
plot_confusion_matrix(mat, class_names=class_dict.values())
plt.savefig(f'res/{save_name}_confusion.png')
plt.show()