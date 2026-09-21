import jax
import jax.numpy as jnp
import flax.linen as nn

STEM_POOL_KERNEL = (3, 3)
STEM_POOL_STRIDE = (2, 2)
STEM_POOL_PADDING = ((1, 1), (1, 1))
# flax convolutions take channels last
SPATIAL_AXES = (1, 2)
TORCH_CONV_TO_FLAX = (2, 3, 1, 0)
UNUSED_TORCH_KEYS = ("num_batches_tracked",)


def conv_spec(conv):
    return dict(
        features=conv.out_channels,
        kernel_size=tuple(conv.kernel_size),
        strides=tuple(conv.stride),
        padding=tuple((pad, pad) for pad in conv.padding),
        kernel_dilation=tuple(conv.dilation),
        feature_group_count=conv.groups,
        use_bias=conv.bias is not None,
    )


def downsample_spec(block):
    if block.downsample is None:
        return None
    return (conv_spec(block.downsample[0]), block.downsample[1].eps)


def batchnorm(eps):
    return nn.BatchNorm(use_running_average=True, epsilon=eps)


def shortcut(downsample, x):
    if downsample is None:
        return x
    return downsample[1](downsample[0](x))


class BasicBlock(nn.Module):
    conv1_spec: dict
    bn1_eps: float
    conv2_spec: dict
    bn2_eps: float
    downsample_spec: tuple | None

    def setup(self):
        self.conv1 = nn.Conv(**self.conv1_spec)
        self.bn1 = batchnorm(self.bn1_eps)
        self.conv2 = nn.Conv(**self.conv2_spec)
        self.bn2 = batchnorm(self.bn2_eps)
        self.downsample = None
        if self.downsample_spec is not None:
            spec, eps = self.downsample_spec
            self.downsample = [nn.Conv(**spec), batchnorm(eps)]

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nn.relu(out + shortcut(self.downsample, x))


class Bottleneck(nn.Module):
    conv1_spec: dict
    bn1_eps: float
    conv2_spec: dict
    bn2_eps: float
    conv3_spec: dict
    bn3_eps: float
    downsample_spec: tuple | None

    def setup(self):
        self.conv1 = nn.Conv(**self.conv1_spec)
        self.bn1 = batchnorm(self.bn1_eps)
        self.conv2 = nn.Conv(**self.conv2_spec)
        self.bn2 = batchnorm(self.bn2_eps)
        self.conv3 = nn.Conv(**self.conv3_spec)
        self.bn3 = batchnorm(self.bn3_eps)
        self.downsample = None
        if self.downsample_spec is not None:
            spec, eps = self.downsample_spec
            self.downsample = [nn.Conv(**spec), batchnorm(eps)]

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return nn.relu(out + shortcut(self.downsample, x))


def mirrored_block(block):
    if type(block).__name__ == "Bottleneck":
        return Bottleneck(
            conv_spec(block.conv1),
            block.bn1.eps,
            conv_spec(block.conv2),
            block.bn2.eps,
            conv_spec(block.conv3),
            block.bn3.eps,
            downsample_spec(block),
        )
    return BasicBlock(
        conv_spec(block.conv1),
        block.bn1.eps,
        conv_spec(block.conv2),
        block.bn2.eps,
        downsample_spec(block),
    )


class ResNet(nn.Module):
    conv1_spec: dict
    bn1_eps: float
    layer_specs: tuple
    classes: int

    def setup(self):
        self.conv1 = nn.Conv(**self.conv1_spec)
        self.bn1 = batchnorm(self.bn1_eps)
        self.layer1 = [mirrored_block(block) for block in self.layer_specs[0]]
        self.layer2 = [mirrored_block(block) for block in self.layer_specs[1]]
        self.layer3 = [mirrored_block(block) for block in self.layer_specs[2]]
        self.layer4 = [mirrored_block(block) for block in self.layer_specs[3]]
        self.fc = nn.Dense(self.classes)

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.max_pool(
            x, STEM_POOL_KERNEL, strides=STEM_POOL_STRIDE, padding=STEM_POOL_PADDING
        )
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x)
        return self.fc(x.mean(axis=SPATIAL_AXES))


# flax names a list entry layer1_0 where torchvision says layer1.0
def flax_path(torch_name):
    path = []
    for segment in torch_name.split("."):
        if segment.isdigit():
            path[-1] = f"{path[-1]}_{segment}"
        else:
            path.append(segment)
    return path


def put(tree, path, leaf, value):
    for segment in path:
        tree = tree.setdefault(segment, {})
    tree[leaf] = value


def flax_variables(state_dict):
    params, batch_stats = {}, {}
    for name, value in state_dict.items():
        if name.endswith(UNUSED_TORCH_KEYS):
            continue
        *path, leaf = flax_path(name)
        array = jnp.asarray(value.cpu().numpy())
        if leaf == "weight" and value.ndim == 4:
            put(params, path, "kernel", array.transpose(TORCH_CONV_TO_FLAX))
        elif leaf == "weight" and value.ndim == 2:
            put(params, path, "kernel", array.T)
        elif leaf == "weight":
            put(params, path, "scale", array)
        elif leaf == "bias":
            put(params, path, "bias", array)
        elif leaf == "running_mean":
            put(batch_stats, path, "mean", array)
        elif leaf == "running_var":
            put(batch_stats, path, "var", array)
    return {"params": params, "batch_stats": batch_stats}


def jax_resnet(torchvision_resnet):
    model = ResNet(
        conv_spec(torchvision_resnet.conv1),
        torchvision_resnet.bn1.eps,
        tuple(
            tuple(layer)
            for layer in (
                torchvision_resnet.layer1,
                torchvision_resnet.layer2,
                torchvision_resnet.layer3,
                torchvision_resnet.layer4,
            )
        ),
        torchvision_resnet.fc.out_features,
    )
    variables = flax_variables(torchvision_resnet.state_dict())
    return jax.jit(lambda x: model.apply(variables, x))
