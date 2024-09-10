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

import torch.nn as nn 


class GILoss(nn.Module):
    def __init__(self):
        super(GILoss, self).__init__() 
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(self, pre_img, data_loader):
        loss = 0
        total_num = 0
        for x, y in data_loader:
            batch_size = x.shape[0]
            total_num += batch_size
            y = y / (800 * 1280)    # 数太大，计算一下平均值
            y_pre = (pre_img * x).flatten(1).mean(dim=1)
            loss += self.mse(y, y_pre)
        loss /= total_num
        return loss
        