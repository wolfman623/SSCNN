# import os
# import glob
# import numpy as np
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


transform = T.Compose([
    T.Resize(90),
    T.CenterCrop(90),
    T.RandomHorizontalFlip(),
    T.RandomSizedCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0,0],std=[1,1])
])


class LoadData(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data_image = data
        self.data_label = labels
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.data_image[index]
        label = self.data_label[index]
        if self.transforms:
            image = self.transforms(image)
            label = self.transforms(label)

        return image, label

    def __len__(self):
        return len(self.data_image)

class LoadTestData(Dataset):
    def __init__(self, data, transforms=None):
        self.data_image = data
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.data_image[index]
        if self.transforms:
            image = self.transforms(image)

        return image

    def __len__(self):
        return len(self.data_image)