
import torch
import time
import copy
import torchvision
import ray
from ray import tune
import torch.nn as nn
from losses import FocalLoss
from torch.optim import lr_scheduler
import os
from data import load_data



def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_info = {"loss":{}, "accuracy":{}}
    loss_lists = {x: list() for x in ["train","val"]}
    acc_lists = {x:list() for x in ["train","val"]}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            loss_lists[phase].append(epoch_loss)
            acc_lists[phase].append(epoch_acc.item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    train_info["loss"].update(loss_lists)
    train_info["accuracy"].update(acc_lists)
    return model, train_info



class Resnet(nn.Module):
    """Simple Neural Network"""
    def __init__(self):
        """
        Parameters:
            base_model: Backbone/Pretrained Neural Network
            base_fc_out: Output unit of the base model
            num_units: Number of Input units of the hidden layer
            drop_rate: Dropout rate
            activation: Activation of hidden unit
        """
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.base_model.fc.in_features

        
        for param in self.base_model.parameters():
            param.requires_grad = False        

        
        # num_ftrs = self.base_model.fc.in_features
        # self.base_model = base_model
        # FC will be set as requires_grad=True by default

        else:
            # self.model = copy.deepcopy(self.base_model)
            
            self.base_model.fc = nn.Linear(num_ftrs, 3)
        
    def forward(self, x):
        x = self.base_model(x)
        return x

class VGG16(nn.Module):
    def __init__(self, num_units, drop_rate, activation):
        super().__init__()

        self.base_model = torchvision.models.vgg16(pretrained=True,).features
        avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        flatten = nn.Flatten(1)
        self.base_model = nn.Sequential(
            self.base_model,
            avg_pool,
            flatten
        )
        num_ftrs = 512 * 7 * 7


        self.base_model.fc = nn.Linear(num_ftrs, num_units)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.fc1 = nn.Linear(num_units, 3)
        self.model = nn.Sequential(
            self.base_model,
            activation,
            self.fc1
        )

    def forward(self,x):
        return self.base_model(x)

class PyTorchTrainable(tune.Trainable):

    # def load_info(self, data_loader,model):
    #     self.trainable_data_loader = data_loader
    #     self.trainable_model = model
      

    def setup(self, config):
        """Set the network for training
        
        Parameters
        ----------
        config: Ray config object that contains the hyperparams        
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(self.device)
        # load data
        # self.train_data_loader, self.valid_data_loader = load_data(BATCH_SIZE, NUM_WORKERS)
        
        # self.train_data_loader, self.valid_data_loader = self.trainable_data_loaderp["train"], self.trainable_data_loader["val"]
        self.data_loader, self.class_names, self.dataset_size = load_data(32)
        self.train_data_loader, self.valid_data_loader= self.data_loader["train"], self.data_loader["val"]


        num_units = config.get("hidden_units", 128)                    
        drop_rate = config.get("drop_rate", 0.0)        
        activation = config.get("activation", nn.ReLU())        
        base_model = config.get("base_model","resnet")
        loss = config.get("loss", "cross")
        lr=config.get("learning_rate", 1e-2), 
        momentum=config.get("momentum", 0.9)


        model_mapper = {
            "resnet": Resnet(),
            "vgg": VGG16(num_units,drop_rate, activation)
        }


        self.model = model_mapper[base_model]
        self.model.to(self.device)

        print("runnongmodel", self.model)
        print("lr",lr)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path,"running_model.txt"), "w") as f:
            f.write(self.model.__repr__())


        self.criterion = nn.CrossEntropyLoss() if loss == "cross" else FocalLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum
        )

        with open(os.path.join(dir_path,"running_lossoptim.txt"), "w") as f:
            f.write(self.criterion.__repr__() + "\n" + self.optimizer.__repr__())
        # self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        
        
    def _train_step(self):
        """Single training loop
        """
        # set to the model train mode

        self.model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.train_data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs,1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.dataset_size["train"] ## need to be checked
        epoch_acc = running_corrects.double() / self.dataset_size["train"] ## need to check size

        

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return epoch_loss, epoch_acc.item()


    
    def _test_step(self):
        """Single test loop
        """        

        self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.valid_data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.inference_mode():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs,1)
                loss = self.criterion(outputs, labels)


            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.dataset_size["val"] ## need to be checked
        epoch_acc = running_corrects.double() / self.dataset_size["val"] ## need to check size

        

        print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        # print(type(epoch_acc),"epoch acc type")
        return epoch_loss, epoch_acc.item()
    
    def step(self):
        """Single training step
        """
        train_loss, train_acc = self._train_step()
        test_loss, test_acc  = self._test_step()

        return {
            "trloss": train_loss,
            "tr_acc": train_acc,
            "tst_acc": test_acc,
            "tst_loss": test_loss
        }

    def save_checkpoint(self, dirname):
        """Saves the model
        
        Parameters
        ----------
            dirname: directory to save the model
        """
        checkpoint_path = os.path.join(dirname, "pytorch-resnet50-raytune.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Loads the model
        
        Parameters
        ----------
            checkpoint_path: load the model from this path
        """
        self.model.load_state_dict(torch.load(checkpoint_path))

    def get_model(self):
        return self.model

    

