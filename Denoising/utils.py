import numpy as np
from skimage.util import view_as_windows
from skimage import io, img_as_float
import os
import torch


def extract_patches(img, patch_size=16, stride=8):
    patches = view_as_windows(img, (patch_size, patch_size), step=stride)
    patches = patches.reshape(-1, patch_size, patch_size)
    return patches

def add_gaussian_noise(patches, sigma):
    noisy = patches + np.random.normal(0, sigma / 255., patches.shape)
    return np.clip(noisy, 0., 1.)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def load_image(path):
    return img_as_float(io.imread(path, as_gray=True))

def prepare_data(sigma, image_dir="../grayscale_images", patch_size=16, stride=8):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png'))]
    # only take 1200 images for training
    image_paths = image_paths[:500]
    patches, clean_patches = [], []
    for path in image_paths:
        img = load_image(path)
        ps = extract_patches(img, patch_size, stride)
        noisy = add_gaussian_noise(img, sigma)
        noisy_patch= extract_patches(noisy, patch_size, stride)
        patches.append(noisy_patch)
        clean_patches.append(ps)

    x = np.concatenate(patches).reshape(-1, patch_size**2)
    y = np.concatenate(clean_patches).reshape(-1, patch_size**2)

    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


