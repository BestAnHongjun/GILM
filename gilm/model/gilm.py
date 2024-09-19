# Copyright (c) 2024, School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), 
# Northwestern PolyTechnical University, 
# and Institute of Artificial Intelligence (TeleAI), China Telecom.
#
# Author:   Coder.AN (an.hongjun@foxmail.com)
#           Huasen Chen (chenyifan1@mail.nwpu.edu.cn)
# 
# 
# This software is licensed under the MIT License.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import torch 
import torch.nn as nn 
from typing import Union, List, Tuple

from diffusers.models.autoencoders import AutoencoderKL 

from .patch_embed import PatchEmbed
from .gi_head import GIHead
from .unet import LatentUNet
from gilm.utils import get_config, get_dtype, get_model


class GILM(nn.Module):
    def __init__(
        self,
        vae_dir: str = None,
        use_fp16: bool = True
    ):
        super(GILM, self).__init__()
        assert os.path.exists(vae_dir) and vae_dir is not None
        self.use_fp16 = use_fp16
        self.vae = AutoencoderKL.from_pretrained(vae_dir)
        self.vae.disable_gradient_checkpointing()
        self.unet = LatentUNet()
        if self.use_fp16:
            self.vae = self.vae.half()
            self.unet = self.unet.half()
        # self.head = GIHead(patch_size, llm_config.hidden_size, self.device, self.dtype)
    
    def forward(self, x):
        assert x.shape == (2, 4, 128, 128)
        sample = self.unet(x)
        # sample = sample.to(torch.float)
        image = self.vae.decode(sample)
        print(image.shape)
        
        # batch_size = x.shape[0]
        # num_patch = x.shape[1]
        # target_shape = (x.shape[2], x.shape[3])
        # attention_mask = torch.ones(batch_size, num_patch, device=self.device)
        # x = self.emb(x) + self.pos(attention_mask, 0)
        # for layer in self.layers:
        #     x = layer(x)[0]
        # x = self.head(x, target_shape)
        return x
    
    def gen_init_input(self):
        sample = torch.randn((2, 4, 128, 128), device=next(self.parameters()).device)
        if self.use_fp16:
            sample = sample.half()
        return sample
    