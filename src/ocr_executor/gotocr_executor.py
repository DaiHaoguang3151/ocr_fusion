from typing import List

import numpy as np

from ocr_executor.ocr_executor_base import OCRExecutorBase
from got_ocr.got_text_generator import GOTTextGenerator

_MODEL_NAME = "/home/ubuntu/Projects_ubuntu/GOT_weights/"

class GOTOCRExecutor(OCRExecutorBase):
    def __init__(self, model_name: str = _MODEL_NAME):
        super(GOTOCRExecutor, self).__init__()
        # 模型
        self.text_generator = GOTTextGenerator(model_name)
    
    def _generate(self, images: List[np.ndarray]) -> List:
        """
        批量生成
        """
        results = []
        for image in images:
            # 构建模型输入
            input_dict = {
                "image": image,
                "type": "ocr"
            }
            result = self.text_generator.generate(input_dict)
            results.append(result)
        return results
