import os

from PIL import Image

import torch
from torchvision.transforms import transforms as trans
from torch.utils.data import Dataset


train_transform = trans.Compose([
        trans.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(0.1),
        trans.RandomVerticalFlip(0.1),
        trans.ToTensor(),
        trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)
test_transform = trans.Compose([
    trans.ToTensor(), 
    trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transformer = {"train": train_transform, "test": test_transform}


class CelebADataset(Dataset):
    def __init__(self, base_dir, label_path, transforms=transformer['train']):
        self.base_dir = base_dir
        self.label_path = label_path
        self.transforms = transforms
        self._load_data()
    
    def _load_data(self):
        with open(os.path.join(self.base_dir, self.label_path), 'r') as f:
            data = f.readlines()
            data = [os.path.join(self.base_dir, each.replace('\n', '')) for each in data]
        self.data = data

    def __getitem__(self, index):
        path, liveness_label = self.data[index].split()
        img = Image.open(path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        label = torch.FloatTensor([1, 0] if liveness_label == '1' else [0, 1])

        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = CelebADataset(
        '/home/eco0930_huydl/Desktop/zalo/CelebA_Spoof',
        'metas/intra_test/train_label.txt',
        transformer['train'],
    )