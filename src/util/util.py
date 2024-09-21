import copy
from typing import List, Optional, Union

import numpy as np
import cv2

def save_detection(save_path: str, 
                   image: np.ndarray, 
                   boxes: List, 
                   texts: Union[List[str]],
                   poly: bool = False):
    """
    绘制检测框，poly决定绘制矩形框还是多边形
    """
    new_image = copy.deepcopy(image)

    if not poly:
        for box in boxes:
            cv2.rectangle(new_image, 
                        box[0], 
                        box[1], color=(255, 0, 0))
        
    else:
        for box in boxes:
            cv2.polylines(new_image, [box], isClosed=True, color=(255, 0, 0))

    if texts is not None:
        new_image = add_texts_to_image(new_image, texts)

    cv2.imwrite(save_path, new_image)


def add_texts_to_image(image: np.ndarray, 
                       texts: List[str],
                       font_scale=0.5, 
                       text_color=(0, 0, 0), 
                       background_color=(255, 255, 255)):
    """
    添加预测的文本
    """
    height, width, _ = image.shape
    
    # 计算扩展区域的宽度
    extension_width = width * 2
    
    # 创建一个新的图片，宽度为原图片宽度加上扩展区域的宽度，高度保持不变
    new_image = np.full((height, width + extension_width, 3), background_color, dtype=np.uint8)
    
    # 将原图片粘贴到新图片的左边
    new_image[:, :width, :] = image
    
    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 从上至下添加文字
    sep_x, sep_y = 100, 20    # 文字摆放间距
    start_y = 50              # 每一列的其实高度
    text_y = start_y          # 当前的文字摆放高度
    text_x = width + sep_x    # 当前的文字摆放宽度
    col_text_width = 0        # 当前列的文本的宽度
    for idx, text in enumerate(texts):
        # 计算文本的宽度和高度
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 1)
    
        # 在扩展区域绘制文本
        cv2.putText(new_image, f"{idx}: " + text, (text_x, text_y), font, font_scale, text_color, 1, cv2.LINE_AA)

        col_text_width = max(col_text_width, text_width)

        # 计算文本的位置
        text_y += text_height + sep_y
        if text_y + 50 > height:
            # 另起一列
            text_y = start_y
            text_x += col_text_width + sep_x
            col_text_width = 0
    
    # 返回新的图片
    return new_image


