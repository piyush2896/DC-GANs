from torch import nn

class ConvTranspose2dBNReLu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernerl_size,
                 strides,
                 padding,
                 bias=False):
        super(ConvTranspose2dBNReLu, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernerl_size,
                strides, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.main(X)

class Generator(nn.Module):
    def __init__(self, in_dim, filters, out_channels):
        super(Generator, self).__init__()
        self.generate = nn.Sequential(
            ConvTranspose2dBNReLu(in_dim, filters*8, 4, 1, 0),
            ConvTranspose2dBNReLu(filters*8, filters*4, 4, 2, 1),
            ConvTranspose2dBNReLu(filters*4, filters*2, 4, 2, 1),
            ConvTranspose2dBNReLu(filters*2, filters, 4, 2, 1),
            nn.ConvTranspose2d(filters, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, X):
        return self.generate(X)
