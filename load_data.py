from PIL import Image
from torch.utils import data
from transform import transformer
class Data(data.Dataset):
    def __init__(self,label_list,transforms):
        self.file_list, self.label = self.get_file_list(label_list)
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.transforms(Image.open(self.file_list[index]).convert("RGB"))

        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.file_list)

    def get_file_list(self,label_lists):
        file_list = []
        label_list = []
        for file in label_lists:
            file_info = file.strip("\n").split(" ")
            file_name = file_info[0]
            label = file_info[1]
            file_list.append(file_name)
            label_list.append(int(label))
        return file_list, label_list