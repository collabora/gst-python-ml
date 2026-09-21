import mlx.core as mx
import mlx.nn as nn

STEM_POOL_KERNEL = 3
STEM_POOL_STRIDE = 2
STEM_POOL_PADDING = 1
# mlx convolutions take channels last
SPATIAL_AXES = (1, 2)
TORCH_CONV_TO_MLX = (0, 2, 3, 1)
UNUSED_TORCH_KEYS = ("num_batches_tracked",)


def mirrored_conv(conv):
    return nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )


def mirrored_batchnorm(batchnorm):
    return nn.BatchNorm(batchnorm.num_features, eps=batchnorm.eps)


def mirrored_downsample(block):
    if block.downsample is None:
        return None
    return [mirrored_conv(block.downsample[0]), mirrored_batchnorm(block.downsample[1])]


def shortcut(downsample, x):
    if downsample is None:
        return x
    return downsample[1](downsample[0](x))


class BasicBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv1 = mirrored_conv(block.conv1)
        self.bn1 = mirrored_batchnorm(block.bn1)
        self.conv2 = mirrored_conv(block.conv2)
        self.bn2 = mirrored_batchnorm(block.bn2)
        self.downsample = mirrored_downsample(block)

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nn.relu(out + shortcut(self.downsample, x))


class Bottleneck(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv1 = mirrored_conv(block.conv1)
        self.bn1 = mirrored_batchnorm(block.bn1)
        self.conv2 = mirrored_conv(block.conv2)
        self.bn2 = mirrored_batchnorm(block.bn2)
        self.conv3 = mirrored_conv(block.conv3)
        self.bn3 = mirrored_batchnorm(block.bn3)
        self.downsample = mirrored_downsample(block)

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return nn.relu(out + shortcut(self.downsample, x))


BLOCKS_BY_NAME = {"BasicBlock": BasicBlock, "Bottleneck": Bottleneck}


def mirrored_layer(layer):
    return [BLOCKS_BY_NAME[type(block).__name__](block) for block in layer]


class ResNet(nn.Module):
    def __init__(self, torchvision_resnet):
        super().__init__()
        self.conv1 = mirrored_conv(torchvision_resnet.conv1)
        self.bn1 = mirrored_batchnorm(torchvision_resnet.bn1)
        self.pool = nn.MaxPool2d(
            STEM_POOL_KERNEL, stride=STEM_POOL_STRIDE, padding=STEM_POOL_PADDING
        )
        self.layer1 = mirrored_layer(torchvision_resnet.layer1)
        self.layer2 = mirrored_layer(torchvision_resnet.layer2)
        self.layer3 = mirrored_layer(torchvision_resnet.layer3)
        self.layer4 = mirrored_layer(torchvision_resnet.layer4)
        self.fc = nn.Linear(
            torchvision_resnet.fc.in_features, torchvision_resnet.fc.out_features
        )

    def __call__(self, x):
        x = self.pool(nn.relu(self.bn1(self.conv1(x))))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x)
        return self.fc(x.mean(axis=SPATIAL_AXES))


def mlx_weight(name, value):
    array = mx.array(value.cpu().numpy())
    if value.ndim == 4:
        return array.transpose(*TORCH_CONV_TO_MLX)
    return array


# the attribute names above match torchvision's, so its state dict loads as is
def mlx_resnet(torchvision_resnet):
    model = ResNet(torchvision_resnet)
    weights = [
        (name, mlx_weight(name, value))
        for name, value in torchvision_resnet.state_dict().items()
        if not name.endswith(UNUSED_TORCH_KEYS)
    ]
    model.load_weights(weights, strict=True)
    model.eval()
    return model
