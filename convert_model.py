import numpy as np
import onnxruntime as rt
import torch
from PIL import Image

from src.data_io import transform as trans
from src.model_lib.MiniFASNet import (MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2,
                                  MiniFASNetV2SE)
from src.model_lib.MultiFTNet import MultiFTNet
from src.utility import get_kernel, parse_model_name

# -----------------------------------------------------------------------------------------------#

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}
model_type = "MiniFASNetV1SE"
device = torch.device("cuda")

model = MultiFTNet(
    model_type=MODEL_MAPPING[model_type],
    conv6_kernel=get_kernel(80, 80),
    img_channel=3,
    num_classes=2,
    training=False,
    pre_trained=None,
).to(device)

state_dict = torch.load(
    "/home/namnh1/zalo-challenge/saved_logs/test_10_100/4.0_80x80_MiniFASNetV1SE.pth", map_location=device
)
keys = iter(state_dict)
first_layer_name = keys.__next__()
if first_layer_name.find("module.") >= 0:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name_key = key[7:]
        new_state_dict[name_key] = value
    model.load_state_dict(new_state_dict)
model.eval()

# ---------------------------------------------------------------------------------------------------#


def convert_ONNX(model, save_name):
    x = torch.randn(1, 3, 80, 80, requires_grad=True).to(device)
    torch.onnx.export(
        model,
        x,
        save_name,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


# export_path = "/home/eco0936_namnh/CODE/zalo-challenge/resources/ckpt_onnx/1_80x80_MiniFASNetV1SE.onnx"
# model = rt.InferenceSession(export_path, providers=["CUDAExecutionProvider"])


# def predict_onnx(model, img: torch.Tensor) -> np.ndarray:
#     input_name = model.get_inputs()[0].name
#     output_name = model.get_outputs()[0].name
#     pred_onx = model.run([output_name], {input_name: img.detach().cpu().numpy()})[0]
#     return pred_onx




if __name__ == "__main__":
    convert_ONNX(model,"/home/namnh1/zalo-challenge/saved_logs/test_10_onnx/4.0_80x80_MiniFASNetV1SE.onnx")
    # img = torch.randn((1, 3, 80, 80)).to(device)
    # print(rt.get_device())
    # print(model)
    # pred = predict_onnx(model, img)
    # print(pred)
