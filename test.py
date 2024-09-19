import torch 
import numpy as np
import matplotlib.pyplot as plt

from gilm.model import GILM, GILoss
from gilm.data import SpeckleData

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    target_shape = (448, 448)
    model = GILM(
        ref_model_dir="/data/anhongjun/projects/llm-imaging/models/opt-125m", 
        dtype=torch.float32
    )
    dataset = SpeckleData(
        speckle_imgs_dir="/data/anhongjun/projects/llm-imaging/datasets/010",
        intensity_txt_path="/data/anhongjun/projects/llm-imaging/datasets/cars5.txt",
        target_shape=target_shape,
        cache=True, dtype=torch.float32
    )
    loss_func = GILoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    noise = torch.ones((1, 1, *target_shape), device=torch.device("cuda"), dtype=torch.float32)
    i = 0
    writer = SummaryWriter("/data/anhongjun/projects/llm-imaging/result/llm_log/")
    while True:
        optimizer.zero_grad()
        y_pre = model(noise)
        y_pre, loss = loss_func(y_pre, dataset)
        if i % 1000 == 0:
            x_out=np.array(y_pre.squeeze(0).detach().cpu(),dtype='double')
            x_out = x_out - np.min(x_out)
            x_out = x_out * 255 / np.max(np.max(x_out))
            x_out = x_out.astype('uint8')
            path = f"/data/anhongjun/projects/llm-imaging/result/llm/{i:06d}.jpg"
            plt.imsave(path, x_out, cmap='gray')
            # cv2.imwrite(path, img)
        print(f"Iter:{i+1}  loss:{loss.item()}")
        writer.add_scalar("Loss", loss.item(), i)
        if loss.item() < 0.05:
            break
        loss.backward()
        optimizer.step()
        i = i + 1
