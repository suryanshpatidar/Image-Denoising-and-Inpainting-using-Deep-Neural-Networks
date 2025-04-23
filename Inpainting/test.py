import torch
import numpy as np
from model import StackedSSDA
from utils import load_image, extract_patches, psnr, add_text_overlay
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denoise_image(model, img, img_path, patch_size=16, stride=8):
    noisy_img = add_text_overlay(img_path)
    noisy_patches = extract_patches(noisy_img, patch_size, stride)
    x = noisy_patches.reshape(-1, patch_size**2)
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x)
    out = out.cpu().numpy().reshape(-1, patch_size, patch_size)

    rec = np.zeros_like(img)
    count = np.zeros_like(img)
    h, w = img.shape
    idx = 0
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            rec[i:i+patch_size, j:j+patch_size] += out[idx]
            count[i:i+patch_size, j:j+patch_size] += 1
            idx += 1
    return rec / count, noisy_patches.reshape(-1, patch_size, patch_size)  # Return patches as "noisy_patches"



# def ksvd_denoise(img, sigma, patch_size=16, stride=8, n_components=256):
#     noisy = add_gaussian_noise(img, sigma)
#     patches = extract_patches(noisy, patch_size, stride)
#     flat_patches = patches.reshape(patches.shape[0], -1)  # shape: (n_patches, 256)

#     # Avoid more atoms than patches
#     n_components = min(n_components, flat_patches.shape[0])

#     # Dictionary learning
#     dict_learner = DictionaryLearning(
#         n_components=n_components,
#         transform_algorithm='omp',
#         transform_n_nonzero_coefs=4,
#         fit_algorithm='lars',
#         max_iter=10,
#         verbose=False,
#         n_jobs=-1,
#         random_state=0
#     )
#     dict_learner.fit(flat_patches)
#     D = dict_learner.components_  # shape: (n_atoms, 256)

#     # Sparse coding using learned dictionary
#     omp = OrthogonalMatchingPursuit(n_nonzero_coefs=4)
#     omp.fit(D.T, flat_patches.T)  # alternative: use .transform() directly
#     codes = dict_learner.transform(flat_patches)  # shape: (n_patches, n_atoms)

#     # Reconstruct patches
#     recon_patches = np.dot(codes, D)  # shape: (n_patches, 256)
#     recon_patches = recon_patches.reshape(-1, patch_size, patch_size)

#     # Reconstruct full image by averaging overlapping patches
#     recon = np.zeros_like(img)
#     count = np.zeros_like(img)
#     h, w = img.shape
#     idx = 0
#     for i in range(0, h - patch_size + 1, stride):
#         for j in range(0, w - patch_size + 1, stride):
#             recon[i:i+patch_size, j:j+patch_size] += recon_patches[idx]
#             count[i:i+patch_size, j:j+patch_size] += 1
#             idx += 1

#     denoised = recon / count
#     return denoised, patches  # Return patches as "noisy_patches_ksvd"


if __name__ == "__main__":

    # test on a set of 100 images in testing-images/grayscale-images folder
    # sigma = 50
    input_dim = 16 * 16
    hidden_dims = [512, 256]
    ssda = StackedSSDA(input_dim, hidden_dims).to(device)
    ssda.load_state_dict(torch.load("model/ssda_inpainting.pth", map_location=device))
    ssda.eval()

    # test_dir = '../testing-images/grayscale-images'  # Directory containing test images
    test_dir = '../medical-images'
    test_image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.png'))]
    test_images = [load_image(path) for path in test_image_paths]
    test_images = np.array(test_images)
    print(f'Loaded {len(test_images)} test images of shape {test_images[0].shape}')

    psnr_noisy = []
    psnr_denoised = []
    # kvsd_psnr_noisy = []
    # kvsd_psnr_denoised = []

    for k, test_img in enumerate(test_images):
        denoised, noisy_patches = denoise_image(ssda, test_img, test_image_paths[k])
        # ksvd_denoised, noisy_patches_ksvd = ksvd_denoise(test_img, sigma=sigma)
        noisy_full = np.zeros_like(test_img)
        count = np.zeros_like(test_img)
        patch_size = 16
        stride = 8
        idx = 0
        for i in range(0, test_img.shape[0] - patch_size + 1, stride):
            for j in range(0, test_img.shape[1] - patch_size + 1, stride):
                noisy_full[i:i+patch_size, j:j+patch_size] += noisy_patches[idx]
                count[i:i+patch_size, j:j+patch_size] += 1
                idx += 1
        noisy_full /= count

        psnr_noisy.append(psnr(test_img, noisy_full))
        psnr_denoised.append(psnr(test_img, denoised))
        # kvsd_psnr_noisy.append(psnr(test_img, noisy_patches_ksvd))
        # kvsd_psnr_denoised.append(psnr(test_img, ksvd_denoised))

        # print("PSNR (Noisy):", psnr(test_img, noisy_full))
        # print("PSNR (Denoised):", psnr(test_img, denoised))

        # Save each denoised image, noisy image and original image in same plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(test_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_full, cmap='gray')
        plt.title('Noisy Image')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(denoised, cmap='gray')
        plt.title('Denoised Image')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"result-medical/test_image_{k}.png")
        plt.close()
        # print(f"Saved test_image_{k}.png")


    # Calculate average PSNR values
    avg_psnr_noisy = np.mean(psnr_noisy)
    avg_psnr_denoised = np.mean(psnr_denoised)
    print(f"Average PSNR (Text-overlay): {avg_psnr_noisy:.2f} dB")
    print(f"Average PSNR (Text-removal): {avg_psnr_denoised:.2f} dB")
    # save the average PSNR values to a text file
    with open("result-medical/psnr_values.txt", "w") as f:
        f.write(f"Average PSNR (Text-overlay): {avg_psnr_noisy:.2f} dB\n")
        f.write(f"Average PSNR (Text-removal): {avg_psnr_denoised:.2f} dB\n")
        # f.write(f"Average PSNR (KSV): {np.mean(ksvd_psnr_denoised):.2f} dB\n")
