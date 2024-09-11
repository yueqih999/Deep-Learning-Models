import gzip
import numpy as np
import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images, self.labels = self.load_data(images_path, labels_path)
        self.transform = transform

    def load_data(self, images_path, labels_path):
        with gzip.open(labels_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
            images = images.reshape(len(labels), 28, 28)
        return images, labels
    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = np.array(image, dtype=np.float32) / 255.0
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image)
        return image, label