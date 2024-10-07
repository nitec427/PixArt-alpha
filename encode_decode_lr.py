import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from omegaconf import OmegaConf

import sys
sys.path.append('/kuacc/users/icetin24/super-resolution-work/S3Diff/src')
from my_utils.training_utils import parse_args_paired_training, PairedDataset, degradation_proc
from diffusers import AutoencoderKL
import torch
config = OmegaConf.load('/kuacc/users/icetin24/super-resolution-work/S3Diff/configs/sr.yaml')
weight_dtype = torch.float16
vae = AutoencoderKL.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="vae", revision=None, variant=None, torch_dtype=weight_dtype).to("cuda")

train_dataset = PairedDataset(config.train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
# val_dataset = PairedDataset(config.validation)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

def tensor_to_image(tensor, is_orig = False):
    tensor = tensor.squeeze().detach().cpu()
    if not is_orig:
        tensor = (tensor + 1) / 2
    img = torch.clamp(tensor, 0, 1)
    return ToPILImage()(img)

sample = next(iter(train_dataloader))

x_src, x_tgt, x_ori_size_src = degradation_proc(config, sample, "cuda")
# encode and decode with vae
x_src = x_src.to("cuda").to(weight_dtype)
x_tgt = x_tgt.to("cuda").to(weight_dtype)
x_ori_size_src = x_ori_size_src.to("cuda").to(weight_dtype)

z_src = vae.encode(x_src).latent_dist.sample()
z_tgt = vae.encode(x_tgt).latent_dist.sample()
z_src_ori = vae.encode(x_ori_size_src).latent_dist.sample()
print(z_src.shape, z_tgt.shape, z_src_ori.shape)
x_src_rec = vae.decode(z_src).sample
x_tgt_rec = vae.decode(z_tgt).sample
x_src_ori_rec = vae.decode(z_src_ori).sample
print(x_src_rec.shape, x_tgt_rec.shape, x_src_ori_rec.shape)

def plot_before_after(img_before, img_after, title_before="Before", title_after="After", i=0, is_orig = False):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    print(img_before.shape, img_after.shape)
    # Plot the original image
    axs[0].imshow(tensor_to_image(img_before, is_orig))
    axs[0].set_title(title_before)
    axs[0].axis("off")
    
    # Plot the reconstructed image
    axs[1].imshow(tensor_to_image(img_after, is_orig))
    axs[1].set_title(title_after)
    axs[1].axis("off")
    
    # plt.imsave(f"{i}.png", tensor_to_image(img_after))
    plt.savefig(f"{title_after}_{i}.png")
    
for i in range(2):
    plot_before_after(x_src[i], x_src_rec[i], title_before="src", title_after="src_rec", i=i)
    plot_before_after(x_tgt[i], x_tgt_rec[i], title_before="tgt", title_after="tgt_rec", i=i)
    plot_before_after(x_ori_size_src[i], x_src_ori_rec[i], title_before="ori", title_after="ori_rec", i=i, is_orig=True)