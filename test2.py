import torch 
import numpy as np
import matplotlib.pyplot as plt

from gilm.model import GILoss, GILM
from gilm.data import SpeckleData


if __name__ == "__main__":
    target_shape = (448, 448)
    model = GILM(vae_dir="/data/anhongjun/projects/llm-imaging/models/stable-diffusion-xl-base-1.0/vae").to('cuda')
    dataset = SpeckleData(
        speckle_imgs_dir="/data/anhongjun/projects/llm-imaging/datasets/010",
        intensity_txt_path="/data/anhongjun/projects/llm-imaging/datasets/cars5.txt",
        target_shape=target_shape,
        cache=True, dtype=torch.float32
    )
    inp = model.gen_init_input()
    model(inp)
    # loss_func = GILoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    # noise = torch.ones((1, 1, *target_shape), device=torch.device("cuda"), dtype=torch.float32)
    # for i in range(2000):
    #     optimizer.zero_grad()
    #     y_pre = model(noise)
    #     y_pre, loss = loss_func(y_pre, dataset)
    #     if i % 50 == 0:
    #         x_out=np.array(y_pre.squeeze(0).detach().cpu(),dtype='double')
    #         x_out = x_out - np.min(x_out)
    #         x_out = x_out * 255 / np.max(np.max(x_out))
    #         x_out = x_out.astype('uint8')
    #         path = f"/data/anhongjun/projects/llm-imaging/result/unet/{i:04d}.jpg"
    #         plt.imsave(path, x_out, cmap='gray')
    #         # cv2.imwrite(path, img)
    #     print(f"Iter:{i+1}/2000  loss:{loss.item()}")
    #     loss.backward()
    #     optimizer.step()
