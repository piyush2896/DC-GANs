from torch import nn

class Conv2dBNLeaky(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 bias=False,
                 apply_bn=True):
        super(Conv2dBNLeaky, self).__init__()
        li = [
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias)
        ]

        if apply_bn:
            li.append(nn.BatchNorm2d(out_channels))
        li.append(nn.LeakyReLU(0.2, inplace=True))
        self.main = nn.Sequential(*li)

    def forward(self, X):
        return self.main(X)

class Discriminator(nn.Module):
    def __init__(self, in_dim, filters):
        super(Discriminator, self).__init__()
        self.discriminate = nn.Sequential(
            Conv2dBNLeaky(in_dim, filters, 4, 2, 1, apply_bn=False),
            Conv2dBNLeaky(filters, filters*2, 4, 2, 1),
            Conv2dBNLeaky(filters*2, filters*4, 4, 2, 1),
            Conv2dBNLeaky(filters*2, filters*4, 4, 2, 1),
            nn.Conv2d(filters*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.discriminate(X)
