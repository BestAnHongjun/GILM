import cv2
import torch
from diffusers.models.autoencoders import AutoencoderKL 


if __name__ == "__main__":
    vae = AutoencoderKL.from_pretrained("/data/anhongjun/projects/llm-imaging/models/stable-diffusion-xl-base-1.0/vae")
    image = cv2.imread("/data/anhongjun/projects/llm-imaging/demo.jpg")
    inp = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    latent = vae.encode(inp)
    print(inp.shape)
    print(latent.latent_dist.parameters.shape)
    