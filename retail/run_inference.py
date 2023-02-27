#!/usr/bin/env python3
###########################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###########################################################################
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import numpy as np
import os
import time
import argparse
import random
import warnings
from argparse import Namespace
from yolox.tools.demo import Predictor
import torch
from loguru import logger
from yolox.exp import get_exp
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import habana_frameworks.torch.core as htcore
SKU_CLASS = ('object')
_COLORS = np.array(
        [
            0.000, 1.000, 1.000  
        ]
        ).astype(np.float32).reshape(-1, 3)

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # Add Habana HPU related arguments
    parser.add_argument('--hpu', action='store_true', help='Use Habana HPU for training')
    parser.add_argument("--use_lazy_mode",
                        default='True', type=lambda x: x.lower() == 'true',
                        help='run model in lazy or eager execution mode, default=True for lazy mode')
    parser.add_argument("--hmp", action="store_true", help="Enable HMP")
    parser.add_argument('--hmp-bf16', default='ops_bf16_yolox.txt', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='ops_fp32_yolox.txt', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    parser.add_argument(
        "--test_image_path", default="./test_images/test_1109.jpg", help="path to images or video")
    parser.add_argument("--save_result",
        action="store_true",
        help="whether to save the inference result of image")
    parser.add_argument(
        "--device",
        default="hpu",
        type=str,
        help="device to run our model")
    return parser


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return img
    
    
def visual(output, img_info, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(img, bboxes, scores, cls, cls_conf, SKU_CLASS)
    return vis_res
    
def get_image_list(test_image_path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(test_image_path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            image_names.append(apath)
    return image_names

def main(exp, args):
    """ _COLORS = np.array(
        [
            0.000, 1.000, 1.000  
        ]
        ).astype(np.float32).reshape(-1, 3) """
    if args.save_result:
        os.makedirs(exp.output_dir, exist_ok=True)

    current_time = time.localtime()
     
    model = exp.get_model(args.hpu, args.hmp)
    model.eval()   
        
    logger.info("loading checkpoint")
    ckpt = torch.load(args.ckpt, map_location=torch.device(args.device))
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    predictor = Predictor(model=model, exp=exp, cls_names=SKU_CLASS, device=torch.device(args.device))

    if os.path.isdir(args.test_image_path):
            files = get_image_list(args.test_image_path)
    else:
        files = [args.test_image_path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = visual(outputs[0], img_info, exp.test_conf)
        if args.save_result:
            save_file_name = os.path.join(exp.output_dir, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)

if __name__ == "__main__":
    name='yolox-s'
    exp_file=None
    args = make_parser().parse_args()
    exp = get_exp(exp_file, name)
    exp.merge(args.opts)
    main(exp, args)