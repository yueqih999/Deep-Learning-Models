from diffusers import DDIMPipeline
import matplotlib.pyplot as plt
import torch

model_id = "google/ddpm-cifar10-32"
ddim = DDIMPipeline.from_pretrained(model_id)

timesteps = [100, 50, 10]
fig, axs = plt.subplots(1, len(timesteps), figsize=(20, 8))
torch.manual_seed(42)

for j, steps in enumerate(timesteps):
    torch.manual_seed(42)  
    result = ddim(num_inference_steps=steps, output_type="pil")
    image = result.images[0]
    
    axs[j].imshow(image)
    axs[j].set_title(f"Steps={steps}")
    axs[j].axis('off')

plt.savefig('Diffusion/DDIM/results.png')
plt.show()