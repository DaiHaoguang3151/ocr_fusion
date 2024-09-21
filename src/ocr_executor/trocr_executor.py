from typing import List

import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocr_executor.ocr_executor_base import OCRExecutorBase

DAFAULT_MODEL_PATH = "/home/ubuntu/Projects_ubuntu/TrOCR/trocr_base_handwritten"

class TrOCRExecutor(OCRExecutorBase):
    def __init__(self, model_path: str = DAFAULT_MODEL_PATH):
        super(TrOCRExecutor, self).__init__()
        # ocr模型
        self.model_path = model_path

        self._init()

    def _init(self):
        # 初始化
        if "trocr" in self.model_path:
            self.processor = TrOCRProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        else:
            raise NotImplementedError

    def _generate(self, images: np.ndarray) -> List:
        """
        根据输入图片批量生成文字
        """
        print("ele===============> ", type(images[0]))
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        # generated_ids = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
        generated_ids = self.model.generate(pixel_values)  # TODO
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text