from torchvision import transforms as trans

train_transform = trans.Compose(
    [   
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=(128, 128), scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
    ]
)
test_transform = trans.Compose([trans.ToTensor()])
transformer = {"train": train_transform, "test": test_transform}
