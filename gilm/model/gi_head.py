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


class GIHead(nn.Module):
    def __init__(
        self, 
        patch_size: int = 16, num_features: int = 768,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super(GIHead, self).__init__() 
        device = device if device is not None else torch.device('cuda')
        dtype = dtype if dtype is not None else torch.float
        
        self.proj = nn.Linear(num_features, patch_size * patch_size, device=device, dtype=dtype)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, target_shape: Union[List, Tuple] = (448, 448)):
        x = self.proj(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, target_shape[0], target_shape[1])
        return x
        