import os
import time

import yaml
import numpy as np
import torch

from paddle_ocr.modeling.base_model import BaseModel
from paddle_ocr.util import transform, create_operators, build_post_process
from paddle_ocr.model_args import parse_args

_args = parse_args()
# _args.det_model_path = "/home/ubuntu/Projects_ubuntu/torchocr/src/paddle_ocr/pretrained_models/det_v4/ch_ptocr_v4_det_infer.pth"
# _args.det_yaml_path = "/home/ubuntu/Projects_ubuntu/torchocr/src/paddle_ocr/pretrained_models/det_v4/ch_PP-OCRv4_det_student.yml"
_args.det_model_path = "/home/ubuntu/Projects_ubuntu/torchocr/src/paddle_ocr/pretrained_models/det_v3/ch_ptocr_v3_det_infer.pth"
    

class TextDetector:
    def __init__(self, args=_args, **kwargs):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.model_path = args.det_model_path
        self.yaml_path = args.det_yaml_path
        self.use_gpu = torch.cuda.is_available()

        self.preprocess_op = None
        self.postprocess_op = None
        self.config = None

        self._init()
        self._build_net_and_load_model(**kwargs)


    def _init(self):
        """
        初始化
        """
        self._get_preprocess_op()
        self._get_postprocess_op()
        self._get_network_config()

    def _get_preprocess_op(self):
        """
        """
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': self.args.det_limit_side_len,
                'limit_type': self.args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        self.preprocess_op = create_operators(pre_process_list)

    def _get_postprocess_op(self):
        """
        """
        if self.args.det_algorithm != "DB":
            raise NotImplementedError
        
        postprocess_params = {}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = self.args.det_db_thresh
        postprocess_params["box_thresh"] = self.args.det_db_box_thresh
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = self.args.det_db_unclip_ratio
        postprocess_params["use_dilation"] = self.args.use_dilation
        postprocess_params["score_mode"] = self.args.det_db_score_mode

        self.postprocess_op = build_post_process(postprocess_params)

    def _get_network_config(self):
        """
        获取模型配置
        """
        if self.yaml_path is not None:
            # det v4
            self.config = self._read_network_config_from_yaml()
            return
        
        model_basename = os.path.basename(self.model_path)
        model_name = model_basename.lower()
        if model_name == 'ch_ptocr_v3_det_infer.pth':
            self.config = {'model_type': 'det',
                           'algorithm': 'DB',
                           'Transform': None,
                           'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                           'Neck': {'name': 'RSEFPN', 'out_channels': 96, 'shortcut': True},
                           'Head': {'name': 'DBHead', 'k': 50}}

    def _read_network_config_from_yaml(self):
        with open(self.yaml_path, encoding='utf-8') as f:
            res = yaml.safe_load(f)
        if res.get('Architecture') is None:
            raise ValueError('{} has no Architecture'.format(self.yaml_path))
        return res['Architecture']

    def _build_net_and_load_model(self, **kwargs):
        # 创建网络
        self.net = BaseModel(self.config, **kwargs)
        # 加载模型
        self.net.load_state_dict(torch.load(self.model_path))
        print('model is loaded: {}'.format(self.model_path))
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()


    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.cuda()
            outputs = self.net(inp)

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs['f_geo'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs['f_border'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
            preds['f_tco'] = outputs['f_tco'].cpu().numpy()
            preds['f_tvo'] = outputs['f_tvo'].cpu().numpy()
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds['maps'] = outputs['maps'].cpu().numpy()
        elif self.det_algorithm == 'FCE':
            for i, (k, output) in enumerate(outputs.items()):
                preds['level_{}'.format(i)] = output
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        if (self.det_algorithm == "SAST" and
            self.det_sast_polygon) or (self.det_algorithm in ["PSE", "FCE"] and
                                       self.postprocess_op.box_type == 'poly'):
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        elapse = time.time() - starttime
        return dt_boxes, elapse

