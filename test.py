import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from torchvision import transforms

from crop_images import crop_face


def generate_bbox(frame):
    model = MTCNN()
    detect_res = model.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bbox = detect_res[0]
    bbox = [int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])]
    return bbox


def predict(path_model, input, bbox, scale=2.7):
    model = torch.load(path_model)
    model.eval()
    if isinstance(input, str):
        img = cv2.imread(input)
    else:
        img = input
    w, h, _ = img.shape
    if w != 128 or h != 128:
        img = cv2.resize(img, (128, 128))

    param = {
        "img": img,
        "bbox": bbox,
        "crop_sz": 128,
        "bbox_ext": (scale - 1.0) / 2,
    }
    img_crop = crop_face(**param)
    transformsa = transforms.Compose([transforms.ToTensor()])
    img_crop = transformsa(img_crop)
    img_crop = img_crop[None, :, :]
    img_crop = img_crop.cuda()
    with torch.no_grad():
        result = model(img_crop)
        result = F.softmax(result).detach().cpu().numpy()
    label = np.argmax(result)
    score = result[0][label]
    if label == 1:
        cv2.rectangle(input, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 1), 2)
        return score
    else:
        cv2.rectangle(input, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        return 1 - score


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        try:
            bbox = generate_bbox(frame)
            score = predict(
                path_model="./model_scale_2.7.pth", input=frame, bbox=bbox, scale=2.7
            )
        except:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
