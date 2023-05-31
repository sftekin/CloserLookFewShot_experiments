from torch import nn

__all__ = ['Conv4', "Conv6"]


def conv_block(in_channels: int, out_channels: int, pool=True) -> nn.Module:
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
    if pool:
        block.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return block


class Conv4(nn.Module):
    def __init__(self, num_classes, remove_linear=False):
        super(Conv4, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        if remove_linear:
            self.logits = None
        else:
            self.logits = nn.Linear(1600, num_classes)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        if self.logits is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.logits(x)
            return x, x1

        return self.logits(x)


class Conv6(nn.Module):
    def __init__(self, num_classes, remove_linear=False):
        super(Conv6, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.conv5 = conv_block(64, 64, False)
        self.conv6 = conv_block(64, 64, False)

        if remove_linear:
            self.logits = None
        else:
            self.logits = nn.Linear(1600, num_classes)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)
        if self.logits is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.logits(x)
            return x, x1

        return self.logits(x)

