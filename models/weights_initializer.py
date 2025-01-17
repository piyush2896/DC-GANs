from torch import nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('2dBN') != -1:
        return
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)