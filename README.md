
`Step 1: Extract frame from video`
```bash
python generate.py
```
`Step 2: Crop image to scale`
```bash
python crop_images_2.py
```
```bash

├───scale_1.0
│   └───liveness_face
│       └───datasets
│           └───datasets_train
|   └───label_list.txt
├───scale_2.7
│   └───liveness_face
│       └───datasets
│           └───datasets_train
|   └───label_list.txt
└───scale_4
    └───liveness_face
        └───datasets
            └───datasets_train 
    └───label_list.txt
```
`Config params: ./src/default_config.py`

`Step 3: training model`
```bash
python train.py
```


### refers:
- https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md
- https://github.com/Elroborn/Face-anti-spoofing-based-on-color-texture-analysis
- https://github.com/mnikitin/Learn-Convolutional-Neural-Network-for-Face-Anti-Spoofing
