import os, glob

from PIL import Image

import torch
from torchvision.transforms import transforms as trans
from torch.utils.data import Dataset


train_transform = trans.Compose([
        # trans.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
        trans.Resize(size=(224, 224)),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(0.1),
        trans.RandomVerticalFlip(0.1),
        # trans.TenCrop(224),
        # trans.Lambda(lambda crops: torch.stack([trans.ToTensor()(crop) for crop in crops])),
        trans.ToTensor(),
        # trans.Lambda(lambda crops: torch.stack([trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop) for crop in crops])),
        trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_transform = trans.Compose([
    trans.ToTensor(),
    # trans.RandomResizedCrop(size=(224, 224)),
    trans.Resize((224, 224)),
    trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transformer = {"train": train_transform, "test": test_transform}


class ZaloDataset(Dataset):
    def __init__(self, image_path, label_path, transforms=transformer['train']):
        self.image_path = image_path
        self.label_path = label_path
        self.transforms = transforms

        self._load_data()

    def _load_data(self):
        with open(self.label_path, 'r') as f:
            label = f.readlines()
            label = [each.replace('\n', '') for each in label]
            label = [(each.split()[0], each.split()[-1]) for each in label]

        self.path_and_label = label
    
    def __len__(self):
        return len(self.path_and_label)

    def __getitem__(self, index):
        path, label = self.path_and_label[index]
        path = os.path.join(self.image_path, os.path.basename(path))
        img = Image.open(path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)
    
        label = torch.FloatTensor([1, 0] if label == '0' else [0, 1])

        return img, label

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


class OnlyFakeDataset(Dataset):
    def __init__(self, base_dir, transforms=transformer['train']):
        self.base_dir = base_dir
        self.transforms = transforms
        self.image_paths = glob.glob(os.path.join(self.base_dir, '**/*.png'), recursive=True)        

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        label = torch.FloatTensor([1, 0])

        return img, label

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    # dataset = CelebADataset(
    #     '/home/eco0930_huydl/Desktop/zalo/CelebA_Spoof',
    #     'metas/intra_test/train_label.txt',
    #     transformer['train'],
    # )

    # dataset = ZaloDataset(
    #     image_path='/home/ai/datasets/challenge/liveness/generate/zalo_train',
    #     label_path='/home/ai/datasets/challenge/liveness/generate/zalo_train/face_crops.txt',
    #     transforms=transformer['train']
    # )
    # dataset = OnlyFakeDataset(
    #     '/home/ai/datasets/challenge/liveness/fake_videos'
    # )

    summary = 21539+8691+23866
    print((21539+8691)/summary)
    print((23866)/summary)