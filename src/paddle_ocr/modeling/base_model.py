from typing import Dict
import torch.nn as nn

from paddle_ocr.modeling.backbones.rec_lcnetv3 import PPLCNetV3
from paddle_ocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3
from paddle_ocr.modeling.necks.db_fpn import RSEFPN
from paddle_ocr.modeling.heads.det_db_head import DBHead

class BaseModel(nn.Module):
    def __init__(self, config: Dict, **kwargs):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()

        in_channels = config.get('in_channels', 3)

        self.use_transform = False
        self.transform = None

        # build backbone, backbone is need for del, rec and cls
        self.use_backbone = True
        config["Backbone"]['in_channels'] = in_channels
        
        backbone_config = config["Backbone"]
        backbone_name = backbone_config.pop("name")
        assert backbone_name in ["PPLCNetV3", "MobileNetV3"]
        if backbone_name == "PPLCNetV3":       # det v4
            self.backbone = PPLCNetV3(**backbone_config)
        else:                                  # det v3
            self.backbone = MobileNetV3(**backbone_config)
        in_channels = self.backbone.out_channels

        # build neck
        self.use_neck = True
        config['Neck']['in_channels'] = in_channels

        neck_config = config["Neck"]
        neck_name = neck_config.pop("name")
        assert neck_name == "RSEFPN"
        self.neck = RSEFPN(**neck_config)           
        in_channels = self.neck.out_channels

        # build head, head is need for det, rec and cls
        self.use_head = True
        config["Head"]['in_channels'] = in_channels

        head_config = config["Head"]
        head_name = head_config.pop("name")
        assert head_name == "DBHead"
        self.head = DBHead(**head_config, **kwargs)

        self.return_all_feats = config.get("return_all_feats", False)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"
        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"
        if self.use_head:
            x = self.head(x)
        # for multi head, save ctc neck out for udml
        if isinstance(x, dict) and 'ctc_nect' in x.keys():
            y['neck_out'] = x['ctc_neck']
            y['head_out'] = x
        elif isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        if self.return_all_feats:
            if self.training:
                return y
            elif isinstance(x, dict):
                return x
            else:
                return {final_name: x}
        else:
            return x