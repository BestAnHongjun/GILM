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

from diffusers.models.activations import get_activation
from diffusers.models.unets.unet_2d_blocks import (
    DownBlock2D, 
    AttnDownBlock2D,
    UNetMidBlock2DCrossAttn,
    AttnUpBlock2D,
    UpBlock2D
)


class LatentUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        norm_eps: float = 1e-5, 
        act_fn: str = 'silu', 
        dropout: float = 0.0
    ):
        super(LatentUNet, self).__init__()
        
        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, 320, kernel_size=conv_in_kernel, padding= conv_in_padding
        )
        
        # down
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(DownBlock2D(
            in_channels=320, out_channels=320, temb_channels=None, dropout=dropout, num_layers=2, 
            resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=32, add_downsample=True, downsample_padding=1, 
        ))
        self.down_blocks.append(AttnDownBlock2D(
            in_channels=320, out_channels=640, temb_channels=None, dropout=dropout, num_layers=2,
            resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=32, attention_head_dim=5, downsample_type='conv', downsample_padding=1
        ))
        self.down_blocks.append(AttnDownBlock2D(
            in_channels=640, out_channels=1280, temb_channels=None, dropout=dropout, num_layers=10, 
            resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=32, attention_head_dim=10, downsample_type=None
        ))
        
        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=1280, out_channels=1280, temb_channels=None, dropout=dropout, num_layers=1,
            transformer_layers_per_block=10, resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=32, num_attention_heads=20
        )
        
        # up
        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(AttnUpBlock2D(
            in_channels=1280, out_channels=1280, prev_output_channel=1280, temb_channels=None, dropout=dropout, num_layers=10, 
            resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=32, attention_head_dim=10, upsample_type='conv', resolution_idx=0
        ))
        self.up_blocks.append(AttnUpBlock2D(
            in_channels=640, out_channels=640, prev_output_channel=1280, temb_channels=None, dropout=dropout, num_layers=2,
            resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=32, attention_head_dim=5, upsample_type='conv', resolution_idx=1 
        ))
        self.up_blocks.append(UpBlock2D(
            in_channels=320, out_channels=320, prev_output_channel=640, temb_channels=None, dropout=dropout, num_layers=2, 
            resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=32, add_upsample=False, resolution_idx=2
        ))
        
        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=320, num_groups=32, eps=norm_eps
        )
        self.conv_act = get_activation(act_fn)
        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            in_channels=320, out_channels=out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )
        
    
    def forward(self, sample: torch.Tensor):
        # 1. pre-process
        sample = self.conv_in(sample)
        
        # 2. down
        down_block_res_samples = (sample, )
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(sample)
            if hasattr(downsample_block, "downsamplers") and downsample_block.downsamplers is not None:
                res_samples = res_samples[:-1]
            down_block_res_samples += res_samples
        
        # 3. mid
        sample = self.mid_block(sample)
        
        # 4. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(hidden_states=sample, res_hidden_states_tuple=res_samples)
        
        # 5. port-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
    
        return sample


if __name__ == "__main__":
    model = LatentUNet()
    sample = torch.zeros((2, 4, 128, 128))
    output = model(sample)
    