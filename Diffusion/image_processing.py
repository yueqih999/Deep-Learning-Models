import numpy as np
import torch
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

def load_images(img_path):
    images = []
    img_paths = [os.path.join(img_path, file) for file in os.listdir(img_path) if file.endswith('.jpg')]

    for path in img_paths:
        with Image.open(path) as img:
            img = img.convert('RGB')
            img_tensor = transforms.ToTensor()(img)
            # img_tensor = transforms.Lambda(lambda t: (t * 2) - 1)(img_tensor)
            images.append(img_tensor)
    return images

if __name__ == "__main__":
    img_path = 'DDPM/images'
    images_tensor = load_images(img_path)

    for i, img in enumerate(images_tensor):
        print(f"Image {i+1} shape: {img}")
