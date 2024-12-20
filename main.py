import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator

'''
Get dataloader for training and testing images:
    - root: root directory of the dataset (train, test, etc.)
    - size: size of the images (128x128)
    - batch_size: batch size for training
'''
def get_dataloader(root, size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(root=root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

'''
Training the GAN with the given parameters:
    - device: device to train on (cpu or gpu)
    - epochs: number of epochs to train for
    - train_loader: dataloader for training images
    - g: generator model
    - d: discriminator model
    - optim_g: optimizer for generator
    - optim_d: optimizer for discriminator
    - criterion: loss function for training
    - batch_size: batch size for training
    - z_dim: dimension of the latent vector for generator
'''
def train_gan(device, epochs, train_loader, g, d, optim_g, optim_d, criterion, batch_size, z_dim):
    for epoch in range(epochs):
        # progress bar
        epoch_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{epochs}]', ncols=100)
        for i, (images, _) in enumerate(epoch_bar):
            real_images = images.to(device)
            real_batch_size = real_images.size(0)

            # skip batches that aren't the set size
            # otherwise network will crash since dimensions won't be as expected
            if real_batch_size < batch_size:
                continue

            real_labels = torch.ones(batch_size, 1).to(device)

            '''
            Train discriminator on real images:
            '''
            optim_d.zero_grad()
            output_real = d(real_images)
            d_loss_real = criterion(output_real, real_labels)

            '''
            Train discriminator on fake images:
            '''
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_images = g(z)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            output_fake = d(fake_images.detach())
            d_loss_fake = criterion(output_fake, fake_labels)

            '''
            Calculate total discriminator loss and update weights:
            '''
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optim_d.step()

            '''
            Train generator:
            '''
            optim_g.zero_grad()
            output_fake = d(fake_images)
            g_loss = criterion(output_fake, real_labels)
            g_loss.backward()
            optim_g.step()

            # update progress bar
            epoch_bar.update(1)
            epoch_bar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

    torch.save(g.state_dict(), 'generator.pth')
    torch.save(d.state_dict(), 'discriminator.pth')

'''
Generate an image using the trained generator:
    - device: device to generate on (cpu or gpu)
'''
def generate_img(device):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('generator.pth'))
    generator.eval()

    # random latent vector
    z = torch.randn(1, 128, 1, 1).to(device)

    # generate an image
    with torch.no_grad():
        generated_image = generator(z)

    generated_image = (generated_image + 1) / 2  # scale pixel value range to [0, 1] for matplot

    grid = make_grid(generated_image, nrow=1, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = False

    if not train:
        generate_img(device)
    else:
        # edit configuration here
        config = {
            'batch_size': 64,
            'z_dim': 128,
            'learning_rate': 0.0002,
            'epochs': 400,
        }

        z = torch.randn(config['batch_size'], config['z_dim'], 1, 1).to(device)

        g = Generator(z=config['z_dim']).to(device)
        d = Discriminator().to(device)

        optim_g = torch.optim.Adam(g.parameters(), lr=config['learning_rate'])
        optim_d = torch.optim.Adam(d.parameters(), lr=config['learning_rate'])

        criterion = torch.nn.BCELoss()

        train_loader = get_dataloader('train', size=128, batch_size=config['batch_size'])
        test_loader = get_dataloader('test', size=128, batch_size=config['batch_size'])

        train_gan(device,
                epochs=['epochs'],
                train_loader=train_loader,
                g=g,
                d=d,
                optim_g=optim_g,
                optim_d=optim_d,
                criterion=criterion,
                batch_size=config['batch_size'],
                z_dim=config['z_dim'])

        generate_img(device)