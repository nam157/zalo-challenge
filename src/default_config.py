# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
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
    conf.pre_trained = "/home/ai/challenge/Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    conf.label_list = "/home/ai/challenge/datasets/crops_80x80/scale_1.0/file_list.txt"
    # ----------------------training---------------
    conf.lr = 1e-3
    # [9, 13, 15]
    conf.milestones = [5, 10, 19, 25]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 300
    conf.momentum = 0.9
    conf.batch_size = 32

    # model
    conf.num_classes = 3
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset
    conf.train_root_path = "./datasets/rgb_image"

    # save file path
    conf.snapshot_dir_path = "./saved_logs/ckpt/test/"

    # log path
    conf.log_path = "./saved_logs/jobs"
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 30

    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = (
        "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"
    )

    # resize fourier image size
    conf.ft_height = 2 * conf.kernel_size[0]
    conf.ft_width = 2 * conf.kernel_size[1]
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    job_name = "Anti_Spoofing_{}".format(args.patch_info)
    log_path = "{}/{}/{} ".format(conf.log_path, job_name, current_time)
    name_ckpt = f"2.7_{w_input}x{h_input}_{conf.model_type}"
    snapshot_dir = "{}/{}".format(conf.snapshot_dir_path, name_ckpt)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf