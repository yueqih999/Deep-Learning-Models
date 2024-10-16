import torch
from torchvision.utils import save_image
from model1 import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_images(generator, model, label, num_images):
    generator.load_state_dict(torch.load(f'{model}1_generator.pth'))
    generator.eval()
    noise = torch.randn(num_images, 100).to(device)
    labels = torch.full((num_images,), label, dtype=torch.long).to(device)
    onehot_labels = torch.zeros(num_images, 10).to(device)
    onehot_labels.scatter_(1, labels.unsqueeze(1), 1)
    fake_imgs = generator(noise, onehot_labels)
    save_image(fake_imgs.data, f'results/{model}1_{label}_generated.png', nrow=1, normalize=True)

if __name__ == '__main__':
    generator = Generator().to(device)
    generate_images(generator, 'dcgan', 0, 1)
    generate_images(generator, 'dcgan', 3, 1)

    generate_images(generator, 'wgan', 0, 1)
    generate_images(generator, 'wgan', 3, 1)

    generate_images(generator, 'wgan_gp', 0, 1)
    generate_images(generator, 'wgan_gp', 3, 1)