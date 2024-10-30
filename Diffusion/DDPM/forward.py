from image_processing import load_images
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def forward_diffusion(img, t, alpha_bar):
    noise = torch.randn_like(img)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t])
    noisy_img = sqrt_alpha_bar_t * img + torch.sqrt(1 - alpha_bar[t]) * noise
    return noisy_img

if __name__ == "__main__":
    timesteps = [0, 10, 50, 100, 500]
    img_path = 'Diffusion/DDPM/images'
    images_tensor = load_images(img_path)
    max_timestep = max(timesteps) 
    beta_schedule = torch.linspace(0.0001, 0.02, max_timestep + 1)
    alpha = 1 - beta_schedule
    alpha_bar = torch.cumprod(alpha, dim=0)


    for i, image in enumerate(images_tensor):
        fig, axs = plt.subplots(1, len(timesteps) + 1, figsize=(12, 4))
        axs[0].imshow(image.permute(1, 2, 0).numpy())
        axs[0].set_title('Original')

        for j, t in enumerate(timesteps):
            noise_image = forward_diffusion(image, t, alpha_bar)
            mse = F.mse_loss(noise_image, image).item()
            noise_image_clamped = torch.clamp(noise_image, 0, 1)
            axs[j+1].imshow(noise_image_clamped.permute(1, 2, 0).numpy())
            axs[j+1].set_title(f"Timestep={t}\nMSE={mse:.4f}")

        plt.suptitle(f'Image {i+1}')
        plt.savefig(f'Diffusion/DDPM/results/noise_image_{i+1}.png')
        plt.close(fig)