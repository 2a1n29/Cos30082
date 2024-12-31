import os
import re
from typing import Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

#Add a label map
def add_label_map(image_path):
    label_map = {}
    with open(image_path, 'r') as file:
        for line in file:
            image_name, label_id = line.strip().split(' ')
            class_name = re.sub(r'(_\d+)?\.jpg$', '', image_name)
            class_name = class_name.replace('_', ' ')
            label_id = int(label_id)
            if label_id not in label_map:
                label_map[label_id] = class_name
    return {k : re.sub(r'\d+$', '', v).strip() for k, v in label_map.items()}

#Visualising samples
def visualise_sample(dataset, number_of_samples=10):
    plt.figure(figsize=(25, 4))
    for i in range(number_of_samples):
        ax = plt.subplot(1, number_of_samples, i + 1)
        img, label = dataset[i] 
        img = img.numpy().transpose((1, 2, 0)) 
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]) 
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f'The label is: {label}\nThe class is: {dataset.label_map[label]}')
        plt.axis('off')
    plt.show()
    
#BirdDataset processor
class BirdDatasetLoader(Dataset):
#Constructor
    def __init__(self, 
                 dataset_path: str, 
                 annotation_file: str, 
                 label_map: Dict[int, str],
                 transform: transforms.Compose = transforms.Compose([
                     transforms.Resize((224, 224)),  
                     transforms.ToTensor(),         
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                 ]),
                 debug: bool = False) -> None:
        self.dataset_path = dataset_path
        self.annotation_file = annotation_file
        self.label_map = label_map
        self.transform = transform
        self.debug = debug
        self.images, self.labels = self.load_dataset()

#Return the length of images
    def __len__(self) -> int:
        return len(self.images)

#Get image and its label
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = os.path.join(self.dataset_path, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        bird_name = self.label_map[label]
        
        if self.debug:
            print(f'Image path is: {image_path}, The label is: {label}, The name of the bird is: {bird_name}')
    
        if self.transform is not None:
            if self.debug:
                print(f'Transforming: {self.transform}')
            image = self.transform(image)
        
        return image, label

#Load the dataset
    def load_dataset(self) -> Tuple[List[str], List[int]]:
        images, labels = [], []
        with open(self.annotation_file, 'r') as file:
            for line in file:
                image_name, label = line.strip().split(' ')
                images.append(image_name)
                labels.append(int(label))
                
        if self.debug:
            print(f'Number of loaded images: {len(images)}')
            print(f'Number of loaded labels: {len(labels)}')
                
        return images, labels