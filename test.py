import cv2
import torch 
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from gilm.model import GILM, GILoss
from gilm.data import SpeckleData


if __name__ == "__main__":
    model = GILM(
        ref_model_dir="/data/anhongjun/projects/llm-imaging/models/opt-125m", 
        dtype=torch.float32
    )
    dataset = SpeckleData(
        speckle_imgs_dir="/data/anhongjun/projects/llm-imaging/datasets/010",
        intensity_txt_path="/data/anhongjun/projects/llm-imaging/datasets/cars5.txt",
        dtype=torch.float32
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    loss_func = GILoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    noise = torch.randn((1, 1, 800, 1280), device=torch.device("cuda"), dtype=torch.float32)
    for i in range(1000):
        optimizer.zero_grad()
        y_pre = model(noise)
        loss = loss_func(y_pre, loader)
        if i % 10 == 0:
            img = y_pre.detach().cpu()[0].squeeze(0).numpy().astype(np.uint8)
            path = f"/data/anhongjun/projects/llm-imaging/result/{i:04d}.jpg"
            cv2.imwrite(path, img)
        print(f"Iter:{i+1}/1000  loss:{loss.item()}")
        loss.backward()
        optimizer.step()
    # img = torch.zeros((1, 1, 1280, 800)).cuda().half()
    # y = model(img)
    # print(img.shape)
    # print(y.shape)