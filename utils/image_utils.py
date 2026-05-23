import torch

def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    mse_clamped = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(1.0 / torch.sqrt(mse_clamped))
