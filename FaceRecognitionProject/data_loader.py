import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

BASE_PATH = './dataset/classification_data'
TRAIN_PATH = os.path.join(BASE_PATH, 'train_data')  
VAL_PATH = os.path.join(BASE_PATH, 'val_data')  
TEST_PATH = os.path.join(BASE_PATH, 'test_data')

class FaceVerificationDataset(Dataset):
    """
    Face Verification dataset for face verification
    
    Methods:
    - __init__: Initialize the dataset
    - __len__: Get the number of samples in the dataset
    - __getitem__: Get a sample from the dataset
    
    Args:
    - root: Root directory of the dataset
    - transform: Transformations to apply to the images
    - max_samples: Maximum number of samples to use from the dataset
    """
    def __init__(self, root, transform=None, max_samples=None):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_paths = []
        self.label_map = {}
        self.label_list = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        for label in self.label_list:
            class_dir = os.path.join(self.root, label)
            class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('jpg', 'jpeg', 'png'))]
            self.label_map[label] = class_images
            self.image_paths.extend(class_images)

        if max_samples is not None:
            self.image_paths = random.sample(self.image_paths, min(max_samples, len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        anchor_path = self.image_paths[index]
        anchor_label = os.path.basename(os.path.dirname(anchor_path))
        anchor = self._load_image(anchor_path)

        # Positive example (same class)
        positive_path = random.choice(self.label_map[anchor_label])
        positive = self._load_image(positive_path)

        # Negative example (different class)
        negative_label = random.choice([label for label in self.label_list if label != anchor_label])
        negative_path = random.choice(self.label_map[negative_label])
        negative = self._load_image(negative_path)

        return anchor, positive, negative

    def _load_image(self, path):
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Example usage
if __name__ == '__main__':
    max_samples = None
    train_dataset = FaceVerificationDataset(TRAIN_PATH, max_samples=max_samples)
    val_dataset = FaceVerificationDataset(VAL_PATH, max_samples=max_samples)
    test_dataset = FaceVerificationDataset(TEST_PATH, max_samples=max_samples)
    
    print(f'----- Original Dataset Information -----')
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    
    max_samples = 1000

    train_dataset = FaceVerificationDataset(TRAIN_PATH, max_samples=max_samples)
    val_dataset = FaceVerificationDataset(VAL_PATH, max_samples=max_samples)
    test_dataset = FaceVerificationDataset(TEST_PATH, max_samples=max_samples)
    
    print(f'----- Dataset Information After Samples -----')
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)
    
    print(f'----- DataLoader Information -----')
    print(f'Train loader size: {len(train_loader)}')
    print(f'Validation loader size: {len(val_loader)}')
    print(f'Test loader size: {len(test_loader)}')

    for i, (anchor, positive, negative) in enumerate(train_loader):
        print(f"Batch {i+1}")
        print(f"Anchor shape: {anchor.shape}, Positive shape: {positive.shape}, Negative shape: {negative.shape}")
        if i == 1:  # Display first two batches for brevity
            break
