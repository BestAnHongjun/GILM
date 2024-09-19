import torch
import argparse
from diffusers import DiffusionPipeline


def make_parser():
    parser = argparse.ArgumentParser("Diffusers SDXL parser.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--shape", type=str, default="1024,1024")
    return parser


def main(args):
    shape = args.shape.split(",")
    assert len(shape) == 2
    width, height = int(shape[0]), int(shape[1])
    pipeline = DiffusionPipeline.from_pretrained(args.model_dir, torch_dtype=torch.float16)
    pipeline.to("cuda")
    image = pipeline(args.prompt, generator=torch.Generator(device="cuda").manual_seed(args.seed),
                     height=height, width=width).images[0]
    if args.save_path != None:
        image.save(args.save_path)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
