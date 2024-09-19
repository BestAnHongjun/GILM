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


class GILoss(nn.Module):
    def __init__(self):
        super(GILoss, self).__init__() 
        self.mse = nn.MSELoss()
    
    def forward(self, out_x, dataset):
        pattern, y_true = dataset.get_all()
        out_y = nn.functional.conv2d(out_x, pattern, stride=1, padding='valid')
        
        #Normalization
        mean_x = out_x.mean(dim=[0, 1, 2, 3], keepdim=True)
        variance_x = out_x.var(dim=[0, 1, 2, 3], keepdim=True, unbiased=False)
        out_x = (out_x - mean_x) / torch.sqrt(variance_x)

        mean_y = out_y.mean(dim=[0, 1, 2, 3], keepdim=True)
        variance_y = out_y.var(dim=[0, 1, 2, 3], keepdim=True, unbiased=False)
        out_y = (out_y - mean_y) / torch.sqrt(variance_y)
        
        x_pred = out_x.squeeze(0)
        y_pred = out_y.flatten(0)
        
        tv_h = torch.abs(x_pred[:, 1:, :] - x_pred[:, :-1, :]).sum()
        tv_w = torch.abs(x_pred[:, :, 1:] - x_pred[:, :, :-1]).sum()
        loss_TV = (tv_h + tv_w) / (1 * x_pred.shape[1] * x_pred.shape[2])
        
        loss = self.mse(y_pred, y_true) + 0.1 * loss_TV
        return x_pred, loss
        