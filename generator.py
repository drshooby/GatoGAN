import torch

class Generator(torch.nn.Module):
    def __init__(self, z):
        super(Generator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z, 2048, kernel_size=4, stride=1, padding=0), # 2048 features
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1), # 1024 features
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), # 512 features
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 256 features
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 128 features
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1), # 3 features, final RGB image
            torch.nn.Tanh() # pixel values in range [-1, 1]
        )

    def forward(self, x):
        return self.net(x)