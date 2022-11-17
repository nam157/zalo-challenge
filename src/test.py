import argparse
import os
import time
import warnings

import cv2
import numpy as np

from model_test import AntiSpoofPredict
from src.crop_images import CropImage
from src.utils import parse_model_name

warnings.filterwarnings("ignore")


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_name)

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))

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
    if len(os.listdir(model_dir)) > 1:
        value = prediction[0][label] / 2
    else:
        value = prediction[0][label]

    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)

    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color,
        2,
    )
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5 * image.shape[0] / 1024,
        color,
    )

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/ai/challenge/Silent-Face-Anti-Spoofing/resources/test/",
        help="model_lib used to test",
    )
    parser.add_argument(
        "--image_name", type=str, default="./image_F2.jpg", help="image used to test"
    )
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)
