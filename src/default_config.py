"""
default config for training
"""

from datetime import datetime

import torch
from easydict import EasyDict

from src.utility import get_kernel, get_width_height, make_if_not_exist


def get_default_config():
    conf = EasyDict()
    conf.model_type = "MiniFASNetV2"
    conf.pre_trained = "/home/namnh1/zalo-challenge/resources/pre-trained/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"

    # ----------------------training---------------
    conf.optimizer_type = "adamw"
    conf.schedule_type = "CosineAnnealingLR"
    conf.lr =  1e-4
    # [9, 13, 15]
    conf.milestones = [9, 15, 20]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 200
    conf.momentum = 0.9
    conf.batch_size = 512

    # model
    conf.num_classes = 3
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset
    conf.label_list = "/home/namnh1/datasets/dataset-spoofings/slient-spoofings_ver5/scale_1.0/file_list.txt"
    # save file path
    conf.snapshot_dir_path = "./saved_logs/ckpt_7/"

    # log path
    conf.log_path = "./saved_logs/jobs"
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 10

    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = torch.device("cuda")

    # resize fourier image size
    conf.ft_height = 2 * conf.kernel_size[0]
    conf.ft_width = 2 * conf.kernel_size[1]
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    job_name = "Anti_Spoofing_{}".format(args.patch_info)
    log_path = "{}/{}/{} ".format(conf.log_path, job_name, current_time)
    conf.name_ckpt = f"1.0_{w_input}x{h_input}_{conf.model_type}"
    snapshot_dir = "{}/{}".format(conf.snapshot_dir_path, conf.name_ckpt)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf
