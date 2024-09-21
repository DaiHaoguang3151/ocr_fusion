## OCR Fusion: EasyOCR/Tesseract/PaddleOCR/TrOCR/GOT

根据个人实际需求尝试了几种不同的OCR，将他们集结在此repo中。

1. [EasyOCR](https://github.com/JaidedAI/EasyOCR)和[Tesseract](https://github.com/tesseract-ocr/tesseract)是之前较流行的轻量OCR工具；
2. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)效果不错，模型也很轻量；为了能在pytorch下直接使用，[PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)提供了转换好的pytorch模型；
3. [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr)是微软实现的，主要用于单行手写文字的识别；
4. [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)是清华最近开源的模型，支持多种形式的OCR识别，效果不错，推荐尝试。

## 内容

[环境](#环境)

[模型下载](#模型下载)

[DEMO](#DEMO)

## 环境

根据`requirements.txt`中的提示进行环境的安装，如有问题，请参考csdn。

## 模型下载

1. EasyOCR: [Jaided AI: EasyOCR model hub](https://www.jaided.ai/easyocr/modelhub/)

2. PaddleOCR (pytorch): [GitHub - frotms/PaddleOCR2Pytorch: PaddleOCR inference in PyTorch. Converted from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)](https://github.com/frotms/PaddleOCR2Pytorch)

3. TrOCR: [microsoft/trocr-base-handwritten · Hugging Face](https://huggingface.co/microsoft/trocr-base-handwritten)

4. GOT: [stepfun-ai/GOT-OCR2_0 · Hugging Face](https://huggingface.co/stepfun-ai/GOT-OCR2_0)

## DEMO

```bash
cd src

# 如果是使用下载好的model，使用时需要在相应的py文件中修改模型的读取路径，比如：
# ocr_executor/gotocr_executor.py中 _MODEL_NAME = "/home/ubuntu/Projects_ubuntu/GOT_weights/"修改为你自己的本地路径

# 选择你需要的ocr执行器和图片，并且运行以下命令
python main.py
```
