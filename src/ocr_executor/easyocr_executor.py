import math
from typing import List

import numpy as np
import easyocr

from ocr_executor.ocr_executor_base import OCRExecutorBase

class EasyOCRExecutor(OCRExecutorBase):
    def __init__(self):
        super(EasyOCRExecutor, self).__init__()
        self.reader = easyocr.Reader(["en"], download_enabled=False)

    def _generate(self, images: List[np.ndarray]) -> List:
        """
        批量生成
        """
        results = []
        for image in images:
            # 可以传图片或者文件路径，detail=1返回检测结果
            result = self.reader.readtext(image, detail=1)
            # for detection in result:
            #     bbox, text, score = detection
            #     print(f"Text: {text}, BBox: {bbox}, Score: {score}")
            results.append(result)
        return results