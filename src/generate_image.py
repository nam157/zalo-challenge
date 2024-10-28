import glob
import os
import warnings

import cv2
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Directories where pre-cropped frames are stored
data_folders = [
    "/home/namnh1/zalo-challenge/data/data_fine_tune/combine",
]
save_dir = "/home/namnh1/zalo-challenge/data/test"
os.makedirs(save_dir, exist_ok=True)

SKIP_FRAME = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MTCNN(device="cuda" if torch.cuda.is_available() else "cpu")


data = pd.read_csv("/home/namnh1/zalo-challenge/data/labels.csv")
face_file = open(os.path.join(save_dir, "face.txt"), "w")
for i in range(len(data)):
    file = data['video_name'].loc[i]
    label = data['label'].loc[i]
    print(label)
    
    frame = cv2.imread(file)
    torch_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(DEVICE)
    detect_res = model.detect(torch_frame)
    if len(detect_res) > 0:
        file_name = os.path.join(save_dir,f"{os.path.basename(file)}",)
        bbox = detect_res[0]
        try:
            face_file.writelines(
                "%s %d %d %d %d %d\n"
                % (
                    file_name,
                    int(bbox[0][0]),
                    int(bbox[0][1]),
                    int(bbox[0][2]),
                    int(bbox[0][3]),
                    label if label is not None else -1,
                )
            )
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
        cv2.imwrite(file_name, frame)

face_file.close()
