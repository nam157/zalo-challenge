from PIL import Image
from torch.utils import data
from torchvision import transforms as trans

train_transform = trans.Compose(
    [
        # trans.RandomResizedCrop(size=(80, 80), scale=(0.9, 1.1)),
        # trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # trans.RandomRotation(10),
        # trans.RandomHorizontalFlip(),
        trans.ToTensor(),
    ]
)
test_transform = trans.Compose([trans.ToTensor()])
transformer = {"train": train_transform, "test": test_transform}


class Dataset(data.Dataset):
    def __init__(self, label_list, transforms):
        self.file_list, self.label = self.get_file_list(label_list)
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms:
            img = self.transforms(Image.open(self.file_list[index]).convert("RGB"))
        else:
            img = Image.open(self.file_list[index]).convert("RGB")

        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.file_list)

    def get_file_list(self, label_lists):
        file_list = []
        label_list = []
        for file in label_lists:
            file_info = file.strip("\n").split(" ")
            file_name = file_info[0]
            label = file_info[1]
            file_list.append(file_name)
            label_list.append(int(label))
        return file_list, label_list
