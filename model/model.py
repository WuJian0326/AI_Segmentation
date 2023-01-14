from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swinUnet import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self,  img_size=224,patch_size=4, in_chans=3,embadding_dim=96,
                 num_classes=1,depth = [2,2,6,2],num_heads=[3,6,12,24],
                 window_size = 7,mlp_ratio=4.,qkv_bias=True,qk_scale=None,ape=False,
                 patch_norm = True,use_checkpoint="home/student/PycharmProjects/SwinUnet/checkpoint/swin_tiny_patch4_window7_224.pth",drop_rate=0.0,drop_path_rate=0.1,
                 zero_head=False, vis=False):

        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.embadding_dim = embadding_dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.ape = ape
        self.patch_norm = patch_norm
        self.use_checkpoint = use_checkpoint
        self.swin_unet = SwinTransformerSys(img_size=self.img_size,
                                patch_size=self.patch_size,
                                in_chans=self.in_chans,
                                num_classes=self.num_classes,
                                embed_dim=self.embadding_dim,
                                depths=self.depth,
                                num_heads=self.num_heads,
                                window_size=self.window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=self.qkv_bias,
                                qk_scale=self.qk_scale,
                                drop_rate=self.drop_rate,
                                drop_path_rate=self.drop_path_rate,
                                ape=self.ape,
                                patch_norm=self.patch_norm,
                                use_checkpoint=self.use_checkpoint,)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, use_checkpoint):
        pretrained_path = use_checkpoint
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")