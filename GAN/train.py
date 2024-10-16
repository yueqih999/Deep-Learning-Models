import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import torch.utils.data as data
from torchvision.utils import save_image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_data = dsets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = dsets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def wgan_loss(fake_scores, real_scores):
    return torch.mean(fake_scores) - torch.mean(real_scores)

def gradient_penalty(batch_size, discriminator, real_data, fake_data, labels):
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = (epsilon * real_data + (1 - epsilon) * fake_data).requires_grad_(True)

    d_interpolates = discriminator(interpolates, labels)
    grad_outputs = torch.ones(d_interpolates.size()).to(device).requires_grad_(False)
    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=grad_outputs,
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    
def train_dcgan(epochs, train_loader,generator, discriminator, optimizer_G, optimizer_D, architecture):
    dcgan_g_loss = []
    dcgan_d_loss = []
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            real_imgs = imgs.to(device) # batch_size, 1, 32, 32
            labels = labels.to(device) # batch_size
            onehot_labels = torch.zeros(len(labels), 10).to(device)
            onehot_labels.scatter_(1, labels.unsqueeze(1), 1)

            noise = torch.randn(len(imgs), 100).to(device) 
            fake_imgs = generator(noise, onehot_labels)

            real_target = torch.ones(len(imgs), 1).to(device)
            fake_target = torch.zeros(len(imgs), 1).to(device)

            criterion = nn.BCELoss()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_scores = discriminator(real_imgs, onehot_labels) # batch_size, 1
            real_disc_loss = criterion(real_scores, real_target)

            fake_scores = discriminator(fake_imgs.detach(), onehot_labels) # batch_size, 1
            fake_disc_loss = criterion(fake_scores, fake_target)

            d_loss = real_disc_loss + fake_disc_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_scores = discriminator(fake_imgs, onehot_labels)
            g_loss = criterion(fake_scores, real_target)
            g_loss.backward()
            optimizer_G.step()  

            if i % 300 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
                dcgan_g_loss.append(g_loss.item())
                dcgan_d_loss.append(d_loss.item())
        save_image(fake_imgs.data[:25], f"dcgan_images/{epoch+1}.png", nrow=5, normalize=True)
        
    torch.save(generator.state_dict(), f'dcgan{architecture}_generator.pth')
    torch.save(discriminator.state_dict(), f'dcgan{architecture}_discriminator.pth')
    print("DCGAN D Loss", dcgan_d_loss)
    print("DCGAN G Loss", dcgan_g_loss)

def train_wgan(epochs, train_loader, generator, discriminator, optimizer_G, optimizer_D, n_critic, architecture):
    wgan_g_loss = []
    wgan_d_loss = []
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            onehot_labels = torch.zeros(len(labels), 10).to(device)
            onehot_labels.scatter_(1, labels.unsqueeze(1), 1)
            noise = torch.randn(len(imgs), 100).to(device)
            fake_imgs = generator(noise, onehot_labels)

            optimizer_D.zero_grad()
            real_scores = discriminator(real_imgs, onehot_labels)
            fake_scores = discriminator(fake_imgs.detach(), onehot_labels)
            loss_D = wgan_loss(fake_scores, real_scores)

            loss_D.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            if i % n_critic == 0:
                optimizer_G.zero_grad()
                fake_scores = discriminator(fake_imgs, onehot_labels)
                loss_G = -torch.mean(fake_scores)
                loss_G.backward()
                optimizer_G.step()

            if i % 300 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], d_loss: {loss_D.item():.4f}, g_loss: {loss_G.item():.4f}")
                wgan_d_loss.append(loss_D.item())
                wgan_g_loss.append(loss_G.item())

        save_image(fake_imgs.data[:25], f"wgan_images/{epoch+1}.png", nrow=5, normalize=True)
        
    torch.save(generator.state_dict(), f'wgan{architecture}_generator.pth')
    torch.save(discriminator.state_dict(), f'wgan{architecture}_discriminator.pth')
    print("WGAN D Loss", wgan_d_loss)
    print("WGAN G Loss", wgan_g_loss)

def train_wgan_gp(epochs, train_loader, generator, discriminator, optimizer_G, optimizer_D, n_critic, lambda_gp, architecture):
    wgan_gp_g_loss = []
    wgan_gp_d_loss = []
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            real_imgs = imgs.to(device) # batch_size, 1, 28, 28
            labels = labels.to(device) # batch_size
            onehot_labels = torch.zeros(len(labels), 10).to(device)
            onehot_labels.scatter_(1, labels.unsqueeze(1), 1)

            noise = torch.randn(len(imgs), 100).to(device) # batch_size, 100
            fake_imgs = generator(noise, onehot_labels) # batch_size, 1, 32, 32

            optimizer_D.zero_grad()
            real_scores = discriminator(real_imgs, onehot_labels)
            fake_scores = discriminator(fake_imgs.detach(), onehot_labels)
            loss_D = wgan_loss(fake_scores, real_scores)
 
            gp = gradient_penalty(len(imgs), discriminator, real_imgs, fake_imgs, onehot_labels)
            loss_D += lambda_gp * gp
            
            loss_D.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            if i % n_critic == 0:
                optimizer_G.zero_grad()
                fake_imgs = generator(noise, onehot_labels)
                fake_scores = discriminator(fake_imgs, onehot_labels)
                loss_G = -torch.mean(fake_scores)
                loss_G.backward()
                optimizer_G.step()

            if i % 300 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], d_loss: {loss_D.item():.4f}, g_loss: {loss_G.item():.4f}")
                wgan_gp_d_loss.append(loss_D.item())
                wgan_gp_g_loss.append(loss_G.item())
        save_image(fake_imgs.data[:25], f"wgan_gp_images/{epoch+1}.png", nrow=5, normalize=True)
    
    torch.save(generator.state_dict(), f'wgan_gp{architecture}_generator.pth')
    torch.save(discriminator.state_dict(), f'wgan_gp{architecture}_discriminator.pth')
    print("WGAN GP D Loss", wgan_gp_d_loss)
    print("WGAN GP G Loss", wgan_gp_g_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN model with custom parameters')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--model', type=str, default="wgan-gp", choices=['dcgan', 'wgan', 'wgan-gp'], help="Model to train: dcgan, wgan, or wgan-gp")
    parser.add_argument('--architecture', type=int, default=1, help='architecture 1 or 2 of the models')
    args = parser.parse_args()

    train_loader, test_loader = load_data(args.batch_size)

    if args.architecture == 1:
        from model1 import Generator, Discriminator
    elif args.architecture == 2:
        from model2 import Generator, Discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    n_critic = 5  
    lambda_gp = 10
    if args.model == 'dcgan':
        train_dcgan(args.num_epochs, train_loader, generator, discriminator, optimizer_G, optimizer_D, args.architecture)
    elif args.model == 'wgan':
        train_wgan(args.num_epochs, train_loader, generator, discriminator, optimizer_G, optimizer_D, n_critic, args.architecture)
    elif args.model == 'wgan-gp':
        train_wgan_gp(args.num_epochs, train_loader, generator, discriminator, optimizer_G, optimizer_D, n_critic, lambda_gp, args.architecture)
