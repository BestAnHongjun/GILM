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

import torch 
import torch.nn as nn 
from typing import Union, List, Tuple

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .patch_embed import PatchEmbed
from .gi_head import GIHead
from gilm.utils import get_config, get_dtype, get_model

class GILM(nn.Module):
    def __init__(
        self,
        ref_model_dir: str = None,
        ref_model_dtype: torch.dtype = None,
        input_shape: Union[List, Tuple] = (800, 1280),
        patch_size: int = 32,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super(GILM, self).__init__()
        self.device = device if device is not None else torch.device('cuda')
        self.dtype = dtype if dtype is not None else torch.float16
        
        llm_config = get_config(ref_model_dir)
        ref_model_dtype = ref_model_dtype if ref_model_dtype is not None else get_dtype(llm_config.torch_dtype)
        llm = get_model(ref_model_dir, ref_model_dtype, self.device, True).to(self.dtype)
        
        self.emb = PatchEmbed(input_shape, patch_size, llm_config.hidden_size, self.device, self.dtype)
        self.num_patch = self.emb.num_patch
        self.pos = llm.model.decoder.embed_positions
        self.layers = llm.model.decoder.layers
        self.head = GIHead(input_shape, patch_size, llm_config.hidden_size, self.device, self.dtype)
    
    def forward(self, x, attention_mask = None):
        batch_size = x.shape[0]
        attention_mask = (
            torch.ones(batch_size, self.num_patch, device=self.device)
            if attention_mask is None
            else attention_mask
        )
        x = self.emb(x) + self.pos(attention_mask, 0)
        for layer in self.layers:
            x = layer(x)[0]
        x = self.head(x)
        return x
    