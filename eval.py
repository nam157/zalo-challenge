import argparse
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd

from model_test import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings("ignore")


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    # print(model_test)
    image_cropper = CropImage()
    if isinstance(image_name, str):
        image = cv2.imread(image_name)
    else:
        image = image_name

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }

        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        print("Image is Real Face. Score: {:.2f}.".format(value))
        return value
    else:
        print("Image is Fake Face. Score: {:.2f}.".format(1 - value))
        return 1 - value


def main(arg):
    target = {}
    for video_name in os.listdir(args.image_name):
        print("Processing video: " + video_name)
        video_path = os.path.join(args.image_name, video_name)
        cap = cv2.VideoCapture(video_path)
        ls = []
        c = 0
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                # if c % 5 == 0:
                score = test(frame, args.model_dir, args.device_id)
                ls.append(score)
                # c +=1   
            except:
                break
        print(ls)
        target[video_name] = sum(ls) / len(ls)
    df = pd.DataFrame(list(target.items()), columns=["fname", "liveness_score"])
    df.to_csv("predict.csv", index=False, encoding="utf-8", float_format="%.10f")


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id", type=int, default=1, help="which gpu id, [0/1/2/3]"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/ai/challenge/Silent-Face-Anti-Spoofing/resources/test_2/",
        help="model_lib used to test",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="/home/ai/challenge/Silent-Face-Anti-Spoofing/datasets/videos/",
        help="image used to test",
    )
    args = parser.parse_args()
    main(args)