import argparse
import os

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator

LATENT_DIMENSION_SIZE = 128
IMG_SIZE = 128

'''
Get dataloader for training and testing images:
    - root: root directory of the dataset (train, test, etc.)
    - size: size of the images (128x128)
    - batch_size: batch size for training
'''
def get_dataloader(root, size, batch_size) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(root=root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6, pin_memory=True)

'''
Create directory for checkpoints if it doesn't exist:
'''
def create_checkpoints_folder():
    try:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
            print('Created checkpoints directory')
    except OSError:
        print('Error: Creating directory of data')
    else:
        print('Successfully created the directory of data')

'''
Save checkpoint for generator and discriminator:
    - g: generator model
    - d: discriminator model
    - optim_g: optimizer for generator
    - optim_d: optimizer for discriminator
    - filename: name of the checkpoint file to save (default: gan_checkpoint.pth)
'''
def save_checkpoint(g, d, optim_g, optim_d, filename='gan_checkpoint.pt') -> None:
    checkpoint_path = os.path.join('checkpoints', filename)
    torch.save({
        'generator_state_dict': g.state_dict(),
        'discriminator_state_dict': d.state_dict(),
        'optimizer_G_state_dict': optim_g.state_dict(),
        'optimizer_D_state_dict': optim_d.state_dict(),
    }, checkpoint_path)
    tqdm.write(f"Checkpoint saved to {checkpoint_path}")
'''
Attempt to load checkpoint for generator and discriminator:
    - filename: name of the checkpoint file to load
    - g: generator model
    - d: discriminator model
    - optim_g: optimizer for generator
    - optim_d: optimizer for discriminator
'''
def load_checkpoint(filename, g, d, optim_g, optim_d, new_lr_g=None, new_lr_d=None) -> bool:
    try:
        checkpoint = torch.load(filename)
        g.load_state_dict(checkpoint['generator_state_dict'])
        d.load_state_dict(checkpoint['discriminator_state_dict'])
        optim_g.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optim_d.load_state_dict(checkpoint['optimizer_D_state_dict'])
        if new_lr_g is not None:
            for param_group in optim_g.param_groups:
                param_group['lr'] = new_lr_g

        if new_lr_d is not None:
            for param_group in optim_d.param_groups:
                param_group['lr'] = new_lr_d
        return True
    except FileNotFoundError:
        tqdm.write(f"Checkpoint {filename} not found. Training from scratch.")
        return False
    except KeyError as e:
        tqdm.write(f"Checkpoint {filename} is invalid. Missing key: {e}. Training from scratch.")
        return False
    except Exception as e:
        tqdm.write(f"An error occurred while loading checkpoint {filename}: {e}. Training from scratch.")
        return False

'''
Generate an image using the trained generator:
    - device: device to generate on (cpu or gpu)
'''
def generate_img(device, g) -> None:
    g.eval()
    g.to(device)

    # random latent vector
    z = torch.randn(1, LATENT_DIMENSION_SIZE, 1, 1).to(device)

    # generate an image
    with torch.no_grad():
        generated_image = g(z)

    generated_image = (generated_image + 1) / 2  # scale pixel value range to [0, 1] for matplot

    grid = make_grid(generated_image, nrow=1, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()

    g.train()

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
def train_gan(device, epochs, train_loader, g, d, optim_g, optim_d, criterion, batch_size, z_dim, save_interval=50) -> None:
    for epoch in range(epochs):
        # progress bar
        epoch_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{epochs}]', ncols=100)
        for i, (images, _) in enumerate(epoch_bar):
            real_images = images.to(device)

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
            epoch_bar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
            epoch_bar.update(1)

        if epoch != 0 and epoch % save_interval == 0:
            save_checkpoint(g, d, optim_g, optim_d, filename=f'checkpoint_{epoch}.pt')
            generate_img(device, g)

    save_checkpoint(g, d, optim_g, optim_d, filename='final_checkpoint.pt')
    generate_img(device, g)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN with customizable parameters.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='Learning rate for the discriminator.')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Learning rate for the generator.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_checkpoint.pt', help='Checkpoint filename to load, not necessary.')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # device and checkpoint setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_checkpoints_folder()

    # parameter config
    config = {
        'batch_size': args.batch_size,
        'z_dim': LATENT_DIMENSION_SIZE,
        'lr_d': args.lr_d,
        'lr_g': args.lr_g,
        'epochs': args.epochs,
    }

    # create models and optimizers (on successful load these will be overwritten)
    g = Generator().to(device)
    d = Discriminator().to(device)
    optim_g = torch.optim.Adam(g.parameters(), lr=config['lr_g'])
    optim_d = torch.optim.Adam(d.parameters(), lr=config['lr_d'])

    # try to load checkpoint
    if load_checkpoint(args.checkpoint, g, d, optim_g, optim_d):
        tqdm.write(f"Checkpoint {args.checkpoint} loaded successfully!")

    # setup loss function
    criterion = torch.nn.BCELoss()

    # setup data loader
    train_loader = get_dataloader('train', size=IMG_SIZE, batch_size=config['batch_size'])

    # do training
    train_gan(device,
            epochs=config['epochs'],
            train_loader=train_loader,
            g=g,
            d=d,
            optim_g=optim_g,
            optim_d=optim_d,
            criterion=criterion,
            batch_size=config['batch_size'],
            z_dim=config['z_dim'])