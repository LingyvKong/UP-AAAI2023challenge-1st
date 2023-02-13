import torch
import torch.nn as nn
from up.utils.model.normalize import build_norm_layer, build_conv_norm
from up.utils.general.log_helper import default_logger as logger
from up.utils.model.initializer import initialize_from_cfg


__all__ = ['xmnet']


class MBConv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 stride=1, kernel_size=3, normalize={'type': 'solo_bn'}, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_residual = (in_features == out_features and stride == 1)

        self.b1 = None
        if in_features != hidden_features or stride != 1:
            layers_b1 = []
            layers_b1.append(build_norm_layer(in_features, normalize)[1])  # (norm_layer(in_features))
            layers_b1.append(nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1),
                                       stride=1, padding=(0, 0), bias=False))
            layers_b1.append(build_norm_layer(hidden_features, normalize)[1])  # (norm_layer(hidden_features))
            layers_b1.append(nn.ReLU(inplace=True))
            self.b1 = nn.Sequential(*layers_b1)

        layers = []
        padding = kernel_size // 2
        layers.append(nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, padding=padding,
                                groups=hidden_features, stride=stride, bias=False))
        layers.append(build_norm_layer(hidden_features, normalize)[1])  # (norm_layer(hidden_features))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(hidden_features, out_features, kernel_size=(1, 1), padding=(0, 0)))
        layers.append(build_norm_layer(out_features, normalize)[1])  # (norm_layer(out_features))
        self.b2 = nn.Sequential(*layers)

    def forward(self, x):
        input = x
        if self.b1 is not None:
            x = self.b1(x)
        x = self.b2(x)

        if self.use_residual:
            return input + x
        return x


class FusedConv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 normalize={'type': 'solo_bn'}, stride=1, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=kernel_size,
                             padding=kernel_size // 2, bias=False, stride=stride)
        self.norm1 = build_norm_layer(hidden_features, normalize)[1]  # norm_layer(hidden_features)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, padding=0, bias=False)
        # self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.norm2 = build_norm_layer(out_features, normalize)[1]  # norm_layer(out_features)
        self.use_residual = (in_features == out_features and stride == 1)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        if self.use_residual:
            return input + x
        return x


class XMNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 width_mult=1.0,
                 round_nearest=8,
                 input_size=224,
                 input_channel=32,
                 out_layers=[4],
                 out_strides=[16],
                 normalize={'type': 'solo_bn'},
                 initializer=None,
                 **kwargs):
        super(XMNet, self).__init__()
        input_channel = input_channel
        self.width_mult = width_mult
        self.out_strides = out_strides
        self.out_layers = out_layers

        init_stride = [2, 2, 2, 1, 2]
        out_planes = []

        blocks = [FusedConv, FusedConv, FusedConv, FusedConv, MBConv]
        kernel_size = [3, 3, 3, 3, 3]
        channels = [48, 80, 128, 128, 160]
        repeats = [2, 3, 3, 3, 6]
        expansions = [4, 6, 3, 2, 5]

        logger.info(f'>>>>>> blocks: {blocks}')
        logger.info(f'>>>>>> kernel_size: {kernel_size}')
        logger.info(f'>>>>>> channels: {channels}')
        logger.info(f'>>>>>> repeats: {repeats}')
        logger.info(f'>>>>>> expansions: {expansions}')

        # building first layer
        input_channel = self._make_divisible(input_channel * min(1.0, width_mult), round_nearest)

        features = [
            build_conv_norm(
                in_channels,
                input_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                normalize=normalize)]
        out_planes.append(input_channel)

        # stage 0
        cin = self._make_divisible(24 * width_mult)
        features.append(
            FusedConv(input_channel, input_channel, cin, normalize, stride=1, kernel_size=3)
        )
        out_planes.append(input_channel)
        for i in range(5):
            for j in range(repeats[i]):
                cout = channels[i]
                s = init_stride[i] if j == 0 else 1

                block = blocks[i](
                    in_features=cin,
                    hidden_features=cin * expansions[i],
                    out_features=cout,
                    normalize=normalize,
                    stride=s,
                    kernel_size=kernel_size[i]
                )
                cin = cout
                features.append(block)
                out_planes.append(cout)

        # make it nn.Sequential
        self.features = nn.ModuleList(features)
        self.out_planes = [out_planes[idx] for idx in [1, 3, 6, 12, 18]]
        self.out_planes = [self.out_planes[idx] for idx in self.out_layers]

        self.init_weights()
        if initializer is not None:
            initialize_from_cfg(self, initializer)

    def _make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def get_outplanes(self):
        """
        get dimension of the output tensor
        """
        return self.out_planes

    def get_outstrides(self):
        """
        Returns:

            - out (:obj:`int`): number of channels of output
        """

        return torch.tensor(self.out_strides, dtype=torch.int)

    def forward(self, input):
        x = input["image"]
        outs = []
        for idx in range(len(self.features)):
            x = self.features[idx](x)
            if idx in [1, 3, 6, 12, 18]:
                outs.append(x)
        features = [outs[i] for i in self.out_layers]

        return {'features': features, 'strides': self.get_outstrides()}


def xmnet(**kwargs):
    model = XMNet(**kwargs)
    return model
