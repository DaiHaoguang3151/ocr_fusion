import math
from typing import List

import numpy as np


class OCRExecutorBase:
    def __init__(self):
        pass

    def _generate(self, images: List[np.ndarray]) -> List:
        """
        批量生成
        """
        raise NotImplementedError
    
    def execute(self, paths_images: List, batch_size=16):
        """
        执行ocr
        """
        results = []
        num = len(paths_images)
        paths = [ele[0] for ele in paths_images]
        # print("IMAGE PATHS: ", paths)
        images = [ele[1] for ele in paths_images]
        iterations = math.ceil(num / batch_size)
        for iter in range(iterations):
            batch_images = images[iter * batch_size: min((iter + 1) * batch_size, num)]
            batch_paths = paths[iter * batch_size: min((iter + 1) * batch_size, num)]
            
            batch_results = self._generate(batch_images)
            results += batch_results
        print("GENERATED: ", results)
        return results