from typing import List

import numpy as np
import pytesseract

from ocr_executor.ocr_executor_base import OCRExecutorBase

class TesseractOCRExecutor(OCRExecutorBase):
    def __init__(self):
        super(TesseractOCRExecutor, self).__init__()
    
    def _generate(self, images: np.ndarray) -> List:
        """
        根据输入图片批量生成文字
        """
        generated = []
        # for image in images:
        #     result = pytesseract.image_to_string(image, config="--psm 7")   # --psm 7 表示单行识别
        #     generated.append(result)

        for image in images:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            result = []
            num = len(data["level"])
            for i in range(num):
                if not data["level"][i] == 5:
                    continue
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                box = [[x, y], [x + w, y + h]]
                text = data["text"][i]
                conf = data["conf"][i]
                result.append((box, text, conf))

            generated.append(result)
        
        return generated