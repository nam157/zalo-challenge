from torch.utils import data
from torchvision import transforms
from PIL import Image

class Data(data.Dataset):
    def __init__(self,db_dir,is_train):
        self.is_train = is_train
        self.file_list,self.label = self.get_file_list(db_dir)
        if self.is_train:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = self.transforms(Image.open(self.file_list[index]).convert('RGB'))

        label = self.label[index]
        return img,label

    def __len__(self):
        return len(self.file_list)

    def get_file_list(self,db_dir):
        file_list = []
        label_list = []
        for file in open(db_dir + "/file_list.txt", "r"):
            file_info = file.strip("\n").split(" ")
            file_name = file_info[0]
            label = file_info[1]
            file_list.append(file_name)
            label_list.append(int(label))
        return file_list,label_list

if __name__ == '__main__':
    data = Data("G:/zalo_challenge/liveness_face/datasets/crop/scale_1.0/",True)
    img,label = data[0]
    print(img.shape)
    print(label)
