import math
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms as trans

from crop_images import CropImage
from utils import MODEL_MAPPING, get_kernel, parse_model_name


class Detection:
    def __init__(self):
        caffemodel = "/home/eco0936_namnh/CODE/anti_spoofing/ckpt/detection/Widerface-RetinaFace.caffemodel"
        deploy = "/home/eco0936_namnh/CODE/anti_spoofing/ckpt/detection/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(
                img,
                (
                    int(192 * math.sqrt(aspect_ratio)),
                    int(192 / math.sqrt(aspect_ratio)),
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, "data")
        out = self.detector.forward("detection_out").squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = (
            out[max_conf_index, 3] * width,
            out[max_conf_index, 4] * height,
            out[max_conf_index, 5] * width,
            out[max_conf_index, 6] * height,
        )
        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(
            h_input,
            w_input,
        )
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(
            self.device
        )

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find("module.") >= 0:
            self.model = torch.nn.DataParallel(self.model)
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose(
            [
                trans.ToTensor(),
            ]
        )
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result


def test(image, model_dir):
    model_test = AntiSpoofPredict()
    image_cropper = CropImage()

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
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

    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        return value
    else:
        return 1 - value


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = {}
    video_dir = "/home/eco0936_namnh/CODE/anti_spoofing/public_test_2/videos/"
    for video_name in os.listdir(video_dir):
        print(f"-------------Process------------: {video_name}")
        ls = []
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        cout = 0
        while True:
            ret, frame = cap.read()
            try:
                score = test(frame, model_dir="./ckpt/anti_spoof/anti_spoofing/")
                ls.append(score)
            except:
                break
        print(ls)
        target[video_name] = sum(ls) / len(ls)
    df = pd.DataFrame(list(target.items()), columns=["fname", "liveness_score"])
    df.to_csv("predict.csv", encoding="utf-8", index=False, float_format="%.15f")
