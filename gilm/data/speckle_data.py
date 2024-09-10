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
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import Dataset 


class SpeckleData(Dataset):
    def __init__(self, speckle_imgs_dir: str, intensity_txt_path: str, cache: bool = False, device: torch.device = None, dtype: torch.dtype = None):
        """
        @param speckle_imgs_dir (str): 散斑图像所在文件夹路径
        @param intensity_txt_path (str): 单像素传感器记录的强度序列
        @param cache (bool): 是否开启缓存加速(占用内存)
        """
        self.device = device if device is not None else torch.device('cuda')
        self.dtype = dtype if dtype is not None else torch.float16
        self.cache = cache
        self.intensity = self._load_intensity(intensity_txt_path)
        self.imgs = self._load_imgs(speckle_imgs_dir, len(self.intensity))
        assert len(self.imgs) > 0 and len(self.intensity) > 0
    
    def __getitem__(self, index):
        if self.cache:
            img = self.imgs[index]
        else:
            img = cv2.imread(self.imgs[index], 0)
        img = torch.tensor(img, dtype=self.dtype, device=self.device).unsqueeze(0)
        intensity = torch.tensor(self.intensity[index], dtype=self.dtype, device=self.device)
        return img, intensity
    
    def __len__(self):
        return min(len(self.imgs), len(self.intensity))
        
    def _load_imgs(self, imgs_dir, max_num):
        imgs = []
        print("Loading speckle images...")
        for i in tqdm(range(max_num)):
            img_path = os.path.join(imgs_dir, f"{i+1}.png")
            assert os.path.exists(img_path)
            if self.cache:
                img = cv2.imread(img_path, 0)
                imgs.append(img)
            else:
                imgs.append(img_path)
        return imgs

    def _load_intensity(self, txt_path):
        assert txt_path[-4:] == ".txt", f"The intensiry data must be a txt file!"
        intensity = []
        f = open(txt_path, "r")
        while True:
            data = f.readline()
            if not data or not data.strip():
                break 
            assert data.strip().isdecimal(), f"{data} is not a number!"
            data = float(data.strip())
            intensity.append(data)
        return intensity
