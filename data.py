# Data augmentation and normalization for training
# Just normalization for validation
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import  transforms, datasets
import os


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def load_data(batch_size=32, num_workers=4 ):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path,"MYCATI/")
    # data_dir = os.path.join("/home/mtcv/Desktop/milad/rafail","MYCATI/")
    data_transforms = {
        
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(),
            transforms.RandomRotation(15),
            transforms.Pad(10),
            transforms.RandomVerticalFlip(),
            transforms.GaussianBlur(3),

            # transfroms.RandomAdjustSharpness(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    }


    ## for trainval folder

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']}

    # class_names = image_datasets['train'].classes
    #_________________________________________________________________

    # for mycati folder

    image_datasets = {}
    all_dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    image_datasets["train"], image_datasets["val"] = torch.utils.data.random_split(all_dataset, [train_size, test_size])

    image_datasets["train"] = MyDataset(image_datasets["train"],data_transforms["train"])

    image_datasets["val"] = MyDataset(image_datasets["val"],data_transforms["val"])
    class_names = all_dataset.classes

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, class_names, dataset_sizes



