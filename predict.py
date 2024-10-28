import cv2
import numpy as np
import os
import warnings

from src.generate_patches import CropImage
from src.model_test import AntiSpoofPredict
from src.utility import parse_model_name

warnings.filterwarnings("ignore")


def test(video_path, model_dir, output_video_path):
    model_test = AntiSpoofPredict()
    image_cropper = CropImage()
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get bounding box
        image_bbox, bbox_org = model_test.get_bbox(frame)
        prediction = np.zeros((1, 2))

        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))

        label = np.argmax(prediction)
        value = prediction[0][label] / len(os.listdir(model_dir))
        if label == 1 and value > 0.85:
            spoof_confidence = value
            text = f"Label: Real, Confidence: {spoof_confidence:.2f}"
            color = (0, 255, 0)
        else:
            spoof_confidence = 1 - value
            text = f"Label: 'Spoof', Confidence: {spoof_confidence:.2f}"
            color = (0, 0, 255)

        # Visualize: draw bbox and prediction result on the frame
        x1, y1, x2, y2 = map(int, bbox_org)  # assuming get_bbox returns [x1, y1, x2, y2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        print(text)
        # Save the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved at {output_video_path}")

def processing_folder(folder):
    for i, video_name in enumerate(os.listdir(folder)):
        print(video_name)
        save_dir = "/home/namnh1/zalo-challenge/datasets/results"
        os.makedirs(save_dir, exist_ok=True)
        test(os.path.join(folder, video_name), "resources/test_7", f"{save_dir}/output_video_{i}.mp4")

def processing_image(video_path):
    test(video_path, "resources/test_5", "output_video_14.mp4")
    


if __name__ == "__main__":
    # processing_image("/home/namnh1/zalo-challenge/datasets/video-tiktok/VID_20241002_082801.mp4")
    processing_folder("/home/namnh1/zalo-challenge/datasets/video-tiktok")