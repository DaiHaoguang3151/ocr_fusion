import os, sys
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse

def init_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    # parser.add_argument("--ir_optim", type=str2bool, default=True)
    # parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    # parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--warmup", type=str2bool, default=False)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_path", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # # EAST parmas
    # parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    # parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    # parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # # SAST parmas
    # parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    # parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    # parser.add_argument("--det_sast_polygon", type=str2bool, default=False)

    # # PSE parmas
    # parser.add_argument("--det_pse_thresh", type=float, default=0)
    # parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    # parser.add_argument("--det_pse_min_area", type=float, default=16)
    # parser.add_argument("--det_pse_box_type", type=str, default='box')
    # parser.add_argument("--det_pse_scale", type=int, default=1)

    # # FCE parmas
    # parser.add_argument("--scales", type=list, default=[8, 16, 32])
    # parser.add_argument("--alpha", type=float, default=1.0)
    # parser.add_argument("--beta", type=float, default=1.0)
    # parser.add_argument("--fourier_degree", type=int, default=5)
    # parser.add_argument("--det_fce_box_type", type=str, default='poly')

    # # params for text recognizer
    # parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    # parser.add_argument("--rec_model_path", type=str)
    # parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    # parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    # parser.add_argument("--rec_char_type", type=str, default='ch')
    # parser.add_argument("--rec_batch_num", type=int, default=6)
    # parser.add_argument("--max_text_length", type=int, default=25)

    # parser.add_argument("--use_space_char", type=str2bool, default=True)
    # parser.add_argument("--drop_score", type=float, default=0.5)
    # parser.add_argument("--limited_max_width", type=int, default=1280)
    # parser.add_argument("--limited_min_width", type=int, default=16)

    # parser.add_argument(
    #     "--vis_font_path", type=str,
    #     default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'doc/fonts/simfang.ttf'))
    # parser.add_argument(
    #     "--rec_char_dict_path",
    #     type=str,
    #     default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    #                          'pytorchocr/utils/ppocr_keys_v1.txt'))

    # # params for text classifier
    # parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    # parser.add_argument("--cls_model_path", type=str)
    # parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    # parser.add_argument("--label_list", type=list, default=['0', '180'])
    # parser.add_argument("--cls_batch_num", type=int, default=6)
    # parser.add_argument("--cls_thresh", type=float, default=0.9)

    # parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    # parser.add_argument("--use_pdserving", type=str2bool, default=False)

    # # params for e2e
    # parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    # parser.add_argument("--e2e_model_path", type=str)
    # parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    # parser.add_argument("--e2e_limit_type", type=str, default='max')

    # # PGNet parmas
    # parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    # parser.add_argument(
    #     "--e2e_char_dict_path", type=str,
    #     default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    #                          'pytorchocr/utils/ic15_dict.txt'))
    # parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    # parser.add_argument("--e2e_pgnet_polygon", type=bool, default=True)
    # parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # # SR parmas
    # parser.add_argument("--sr_model_path", type=str)
    # parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    # parser.add_argument("--sr_batch_num", type=int, default=1)

    # params .yaml
    parser.add_argument("--det_yaml_path", type=str, default=None)
    parser.add_argument("--rec_yaml_path", type=str, default=None)
    parser.add_argument("--cls_yaml_path", type=str, default=None)
    parser.add_argument("--e2e_yaml_path", type=str, default=None)
    parser.add_argument("--sr_yaml_path", type=str, default=None)

    # # multi-process
    # parser.add_argument("--use_mp", type=str2bool, default=False)
    # parser.add_argument("--total_process_num", type=int, default=1)
    # parser.add_argument("--process_id", type=int, default=0)

    # parser.add_argument("--benchmark", type=str2bool, default=False)
    # parser.add_argument("--save_log_path", type=str, default="./log_output/")

    # parser.add_argument("--show_log", type=str2bool, default=True)

    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()

def get_default_config(args):
    return vars(args)
