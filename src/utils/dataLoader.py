import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class MalpracticeDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_name)
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.jpg', '.json'))
        
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        if self.transform:
            image = self.transform(image)
        
        return image, annotations

def get_dataloader(image_dir, annotation_dir, batch_size=4, shuffle=True, transform=None):
    dataset = MalpracticeDataset(image_dir, annotation_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
