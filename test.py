import os, glob
from tqdm import tqdm
import pandas as pd
import cv2

import torch

from models import MobileNet, mobilevit_s
from load_data import test_transform

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'ckpt/'


if __name__ == '__main__':
    # model = MobileNet()
    model = mobilevit_s((224, 224), 2)
    model = model.to(DEVICE)
    ckpt = torch.load("ckpt_vit_zalo/mobilenet_epoch_0_trainloss_0.686883_validloss_0.685381.pth", map_location=DEVICE)
    model.load_state_dict(ckpt)

    videos = glob.glob("/home/ai/datasets/challenge/liveness/test/public_test_2/videos/*.mp4")
    submit = {}
    # for video in tqdm(videos):
    for video in videos:
        result = torch.zeros(2, device=DEVICE)
        counter = 0
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = test_transform(image)
            image = image.unsqueeze(0)
            image = image.to(DEVICE)

            pred = model(image)
            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = pred.squeeze(0)

            result = result + pred
            
            success, image = vidcap.read()
            counter += 1

        
        result = result / counter
        pos = torch.argmax(result, dim=0)
        ans = 1 - result[pos] if pos == 0 else result[pos]
        submit[os.path.basename(video)] = ans.item()
        break
    csv_file = pd.DataFrame(submit.items(), columns=['Date', 'DateValue'])
    csv_file = csv_file.to_csv('submit.csv')