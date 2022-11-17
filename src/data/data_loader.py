from torch.utils.data import DataLoader
from torchvision import transforms as trans

from src.data.load_data import Dataset


def get_train_loader(conf):
    train_transform = trans.Compose(
        [
            trans.ToPILImage(),
            trans.RandomResizedCrop(size=tuple(conf.input_size), scale=(0.9, 1.1)),
            trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            trans.RandomRotation(10),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
        ]
    )
    trainset = Dataset(
        label_path=conf.train_root_path,
        transforms=train_transform,
        ft_height=conf.ft_height,
        ft_width=conf.ft_width,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
    )
    return train_loader
