import os
from typing import Any, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from torchvision import transforms

from crop_images import crop_face
from crop_images_2 import CropImage
from utils import parse_model_name


def generate_bbox(frame):
    model = MTCNN()
    detect_res = model.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bbox = detect_res[0]
    bbox = [int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])]
    return bbox


def predict(
    model_dir: str,
    input: Any,
    transformer: transforms.Compose,
    bbox: List[int],
    device: torch.device,
) -> int:
    if isinstance(input, str):
        img = cv2.imread(input)
    else:
        img = input
    w, h, channel = img.shape
    if w != 128 or h != 128:
        img = cv2.resize(img, (128, 128))

    prediction = np.zeros((1, 2))
    for model_name in os.listdir(model_dir):
        h_input, w_input, scale = parse_model_name(model_name)
        img_croper = CropImage()
        img_crop = img_croper.crop(org_img=img,bbox=bbox,scale=scale,out_w=w_input,out_h=h_input,crop=True)
        img_transform = transformer(img_crop)
        img_transform = img_transform[None, :, :]
        img_transform = img_transform.to(device)

        path_model = os.path.join(model_dir, model_name)
        model = torch.load(path_model, map_location=device)
        model.eval()
        with torch.no_grad():
            result = model(img_transform)
            result = F.softmax(result, dim=1).detach().cpu().numpy()
        prediction += result
    label = np.argmax(prediction)
    score = prediction[0][label] / 2
    if label == 1:
        cv2.rectangle(input, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 1), 2)
        return label, score
    else:
        cv2.rectangle(input, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        return label, 1 - score


def main(path_img: str = None, is_camera: bool = True):
    transformser = transforms.Compose([transforms.ToTensor()])
    model_dir = "./ckpt/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_camera:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                bbox = generate_bbox(frame)
                score = predict(
                    model_dir=model_dir,
                    input=frame,
                    transformer=transformser,
                    bbox=bbox,
                    device=device,
                )
            except:
                break
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(path_img)
        bbox = generate_bbox(img)
        label, score = predict(
            model_dir=model_dir,
            input=img,
            transformer=transformser,
            bbox=bbox,
            device=device,
        )
        print(label, score)
        cv2.imshow("frame", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main(
        path_img="./datasets/crops/scale_1.0/images_train/datasets/images/1_frame_0.jpg",
        is_camera=True,
    )
