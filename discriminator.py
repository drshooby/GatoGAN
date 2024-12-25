import torch

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),  # 64x64
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.4),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.4),
            torch.nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=1),  # 8x8
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=1),  # 4x4
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(1024, 2048, kernel_size=1, stride=1),  # 2x2
            torch.nn.LeakyReLU(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(51200, 1),  # magic number 51200, if this value is not present everything crashes (sorry.)
            torch.nn.Sigmoid()  # for output between [0, 1], fake or real
        )

    def forward(self, x):
        return self.net(x)