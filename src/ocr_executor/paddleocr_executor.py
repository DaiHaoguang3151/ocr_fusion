from typing import Dict, List

import numpy as np
import cv2
from PIL import ImageFont
from paddleocr import PaddleOCR, draw_ocr

from ocr_executor.ocr_executor_base import OCRExecutorBase

class PaddleOCRExecutor(OCRExecutorBase):
    def __init__(self, lang: str = "en", rec: bool = True):
        super(PaddleOCRExecutor, self).__init__()
        # 识别语言
        self.lang: str = lang
        # 是否识别
        self.rec = rec
        
        self._init()

    def _init(self):
        # 初始化
        self.model = PaddleOCR(use_angle_cls=True, 
                               lang=self.lang, 
                               det=True, 
                               rec=self.rec)

    def _generate(self, images: np.ndarray) -> List:
        """
        根据输入图片生成文字（可选）和返回相应的box
        images可以传入路径
        """
        # 暂时不能批量处理
        generated = []
        for image in images:
            # result = self.model.ocr(image_path, det=True, rec=self.rec, cls=True)[0]
            if not isinstance(image, np.ndarray):
                image_ = np.array(image)   # TODO: 最好看一下通道顺序
                print("image_.shape: ", image_.shape)
            else:
                image_ = image
            result = self.model.ocr(image_, det=True, rec=self.rec, cls=True)[0]
            # self._draw_single_result(image_, result)

            generated.append(result)
        return generated
    
    def _draw_single_result(self, image, result):
        """
        绘制一张图片的检测和识别结果
        """
        # for line in result:
        #     print(line)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        # font = ImageFont.load_default()
        im_show = draw_ocr(image, boxes, txts, scores, font_path="/usr/share/fonts/truetype/ttf-khmeros-core/KhmerOS.ttf")
        cv2.imwrite("./result.jpg", im_show)