import os

import numpy as np
import cv2

# from ocr_executor.easyocr_executor import EasyOCRExecutor
# from ocr_executor.tesseractocr_executor import TesseractOCRExecutor
# from ocr_executor.paddleocr_executor import PaddleOCRExecutor
# from paddle_ocr.text_detector import TextDetector
# from ocr_executor.trocr_executor import TrOCRExecutor
from ocr_executor.gotocr_executor import GOTOCRExecutor
from util.util import save_detection


if __name__ == "__main__":
    # 批量处理
    # image_path = "./images/handwriting.png"
    image_path = "/home/ubuntu/Projects_ubuntu/ocr_fusion/src/images/handwriting.png"
    # image_path = "/home/ubuntu/Projects_ubuntu/ocr_fusion/src/images/handwriting_single_line.png"  # trocr只能识别单行手写文本
    image = cv2.imread(image_path)
    paths_images = [(image_path, image)]

    output_dir = "./images_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1) easyocr
    # easyocr_executor = EasyOCRExecutor()
    # result = easyocr_executor.execute(paths_images)[0]   # 取第一个结果，对应第一张图片
    # bboxes = []
    # texts = []
    # for idx, detection in enumerate(result):
    #     bbox, text, score = detection
    #     bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
    #     bbox = np.array(bbox).astype(np.int32).reshape(-1, 2)
    #     bboxes.append(bbox)
    #     texts.append(text)
    # save_detection(os.path.join(output_dir, "easyocr.png"), image, bboxes, texts=texts, poly=True)

    # 2) tesseractocr
    # tesseractocr_executor = TesseractOCRExecutor()
    # result = tesseractocr_executor.execute(paths_images)[0]
    # bboxes = []
    # texts = []
    # for idx, detection in enumerate(result):
    #     bbox, text, conf = detection
    #     bboxes.append(bbox)
    #     texts.append(text)
    # save_detection(os.path.join(output_dir, "tesseractocr.png"), image, bboxes, texts=texts, poly=False)

    # 3) paddleocr
    # paddleocr_executor = PaddleOCRExecutor()
    # result = paddleocr_executor.execute(paths_images)[0]

    # bboxes = []
    # texts = []
    # for idx, detection in enumerate(result):
    #     bbox, (text, score) = detection
    #     bbox = np.array(bbox).astype(np.int32).reshape(-1, 2)
    #     bboxes.append(bbox)
    #     texts.append(text)
    # save_detection(os.path.join(output_dir, "paddleocr.png"), image, bboxes, texts=texts, poly=True)

    # 4) paddleocr (pytorch det model)
    # text_detector = TextDetector()
    # bboxes, _ = text_detector(image)
    # bboxes = [np.array(bbox).astype(np.int32).reshape((-1, 2)) for bbox in bboxes.tolist()]
    # save_detection(os.path.join(output_dir, "paddleocr_pytorch_det.png"), image, bboxes, texts=None, poly=True)

    # 5) trocr
    # trocr_executor = TrOCRExecutor()
    # text = trocr_executor.execute(paths_images)[0]
    # save_detection(os.path.join(output_dir, "trocr.png"), image, [], texts=[text], poly=True)

    # 6) gotocr
    gotocr_executor = GOTOCRExecutor()
    result = gotocr_executor.execute(paths_images)[0]
    save_detection(os.path.join(output_dir, "gotocr.png"), image, [], texts=[result], poly=True)
