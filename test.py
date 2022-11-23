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
# DEVICE = torch.device('cpu')


if __name__ == '__main__':
    model = MobileNet()
    # model = mobilevit_s((224, 224), 2)
    model = model.to(DEVICE)
    ckpt = torch.load(
        "ckpt_vit/mobilenet_epoch_469_trainloss_0.313851_validloss_0.313995.pth",
        map_location=DEVICE
    )
    model.load_state_dict(ckpt)

    videos = glob.glob("/home/ai/datasets/challenge/liveness/test/public_test_2/videos/*.mp4")
    submit = {}
    # for video in tqdm(videos):
    for video in tqdm(videos):
        result = torch.zeros(0, device=DEVICE)
        counter = 0
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        while success:
            if counter % 10 == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = test_transform(image)
                image = image.unsqueeze(0)
                image = image.to(DEVICE)

                pred = model(image)
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = pred.squeeze(0)

                pos = torch.argmax(pred)
                ans = 1. - pred[pos] if pos == 0 else pred[pos]
                result += ans
                # print(ans)
            
            success, image = vidcap.read()
            counter += 1

        result = result / counter        
        submit[os.path.basename(video)] = round(ans.item(), 5)

    csv_file = pd.DataFrame(submit.items(), columns=['fname', 'liveness_score'])
    csv_file = csv_file.to_csv('submit.csv', index=False)