# Data augmentation and normalization for training
# Just normalization for validation
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import  transforms

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



