import os
from typing import Dict

from PIL import Image
import cv2
import torch
from transformers import AutoTokenizer, TextStreamer

from got_ocr.util.conversation import conv_templates, SeparatorStyle
from got_ocr.util.util import KeywordsStoppingCriteria
from got_ocr.model.GOT_ocr_2_0 import GOTQwenForCausalLM
from got_ocr.model.plug.blip_process import BlipImageEvalProcessor


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


class GOTTextGenerator:
    def __init__(self, model_name: str):
        # 模型名称
        self.model_name = os.path.expanduser(model_name)
        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # 模型
        self.model = GOTQwenForCausalLM.from_pretrained(self.model_name, 
                                                        low_cpu_mem_usage=True, 
                                                        device_map='cuda', 
                                                        use_safetensors=True, 
                                                        pad_token_id=151643).eval()
        self.model.to(device='cuda',  dtype=torch.bfloat16)

        # 图像处理器
        self.image_processor = BlipImageEvalProcessor(image_size=1024)
        self.image_processor_high = BlipImageEvalProcessor(image_size=1024)

    def _get_conv(self, input_dict: Dict):
        """
        处理输入
        """
        # 默认的一些参数
        _default_input_dict = {
            'use_im_start_end': True,
            'conv_mode': 'mpt'
        }
        input_dict.update(_default_input_dict)
        
        
        if input_dict['type'] == 'format':
            qs = 'OCR with format: '
        else:
            qs = 'OCR: '

        h, w, _ = input_dict["image"].shape
        
        if input_dict.get('box'):
            bbox = eval(input_dict['box'])
            if len(bbox) == 2:
                bbox[0] = int(bbox[0]/ w * 1000)
                bbox[1] = int(bbox[1]/ h * 1000)
            if len(bbox) == 4:
                bbox[0] = int(bbox[0]/ w * 1000)
                bbox[1] = int(bbox[1]/ h * 1000)
                bbox[2] = int(bbox[2]/ w * 1000)
                bbox[3] = int(bbox[3]/ h * 1000)
            if input_dict['type'] == 'format':
                qs = str(bbox) + ' ' + 'OCR with format: '
            else:
                qs = str(bbox) + ' ' + 'OCR: '
        
        if input_dict.get('color'):
            if input_dict['type'] == 'format':
                qs = '[' + input_dict['color'] + ']' + ' ' + 'OCR with format: '
            else:
                qs = '[' + input_dict['color'] + ']' + ' ' + 'OCR: '
        
        if input_dict['use_im_start_end']:
            image_token_len = 256
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[input_dict["conv_mode"]].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv

    
    def generate(self, input_dict: Dict):
        """
        根据输入生成相应的文本
        """
        # 图像
        # 输入的image是opencv读取的，需要转成PIL图片
        image_cv2 = input_dict["image"]
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image_rgb)
        image_1 = image.copy()
        image_tensor = self.image_processor(image)
        image_tensor_1 = self.image_processor_high(image_1)

        # 文字
        conv = self._get_conv(input_dict)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = CustomTextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                input_ids,
                images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                do_sample=False,
                num_beams = 1,
                no_repeat_ngram_size = 20,
                streamer=streamer,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria]
                )
            
            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # render: pass，这边用不着
        return outputs
    

class CustomTextStreamer(TextStreamer):
    """
    相较于TextStreamer，唯一的改变就是去掉打印
    """

    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        

