from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as tortrans

from src.data_io import transform as trans
from src.data_io.dataset_folder import Dataset
from src.default_config import get_default_config

# Load default configuration
conf = get_default_config()

# Read labels from the label list file
try:
    with open(conf.label_list, "r") as labels:
        data_label = labels.readlines()
except FileNotFoundError:
    print(f"Label list file not found: {conf.label_list}")
    raise

# Split data into training and validation sets
train_label, val_label = train_test_split(data_label, test_size=0.2, random_state=42)
print(f"Train: {len(train_label)}, Validation: {len(val_label)}")

def get_train_loader(conf):
    # Define training transformations
    train_transform = trans.Compose(
        [
            trans.ToPILImage(),
            trans.RandomResizedCrop(size=tuple(conf.input_size), scale=(0.9, 1.1)),
            trans.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            trans.RandomRotation(10),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
        ]
    )

    # Create training dataset and data loader
    trainset = Dataset(
        label_list=train_label,
        transforms=train_transform,
        ft_width=conf.ft_width,
        ft_height=conf.ft_height,
    )
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
    )
    return train_loader

def get_val_loader(conf):
    # Define validation transformations
    val_transform = trans.Compose(
        [
            trans.ToPILImage(),
            tortrans.Resize(size=tuple(conf.input_size)),
            trans.ToTensor(),
        ]
    )

    # Create validation dataset and data loader
    valset = Dataset(
        label_list=val_label,
        transforms=val_transform,
        ft_width=conf.ft_width,
        ft_height=conf.ft_height,
    )
    val_loader = DataLoader(
        valset,
        batch_size=conf.batch_size,  # Use the same batch size as training for consistency
        shuffle=False,
        pin_memory=True,
        num_workers=16,
    )
    return val_loader
