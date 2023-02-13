# Standard Library
from collections.abc import Iterable

# Import from third library
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

try:
    from up.extensions import DeformableConv
except: # noqa
    DeformableConv = None
from up.utils.model.block_helper import RepConv
from up.utils.model.initializer import initialize_from_cfg
from up.utils.model.normalize import build_norm_layer
from .qarepvgg_block import QARepVGGBlock
__all__ = ['ResNet_qavgg', 'resnet18_qavgg']


deploy = False

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 normalize={'type': 'solo_bn'},
                 stride_in_1x1=False,
                 drop_path=None):
        super(BasicBlock, self).__init__()

        self.drop_path = drop_path
        self.norm1_name, norm1 = build_norm_layer(planes, normalize, 1)
        self.norm2_name, norm2 = build_norm_layer(planes, normalize, 2)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=False)
        self.qablock = QARepVGGBlock(planes, planes)
        self.add_module(self.norm2_name, norm2)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        self.qablock.prepare_for_deploy()
        out = self.qablock(out)
        # out = self.norm2(out)

        # if self.drop_path is not None:
        #     out = self.drop_path(out)
        #
        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)
        return out

def make_layer4(inplanes,
                block,
                planes,
                blocks,
                stride=1,
                dilation=1,
                normalize={'type': 'solo_bn'},
                stride_in_1x1=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(planes * block.expansion, normalize)[1]
        )

    layers = []
    layers.append(block(inplanes, planes, stride, dilation, downsample, normalize, stride_in_1x1))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, stride=1, dilation=dilation,
                            downsample=None, normalize=normalize, stride_in_1x1=stride_in_1x1))

    return nn.Sequential(*layers)

class Block(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 stride = 1,
                 normalize={'type': 'freeze_bn'}):
        super(Block, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.stride = stride


        self.conv = nn.Sequential(
        nn.Conv2d(self.input_channel, self.output_channel, kernel_size=3, stride=self.stride, padding=1, bias=False),
        build_norm_layer(self.output_channel, normalize)[1],
        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DWBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 normalize={'type': 'freeze_bn'}):
        super(DWBlock, self).__init__()

        expand_ratio = 6
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            build_norm_layer(inp * expand_ratio, normalize)[1],
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      1,
                      padding=1,
                      groups=inp * expand_ratio,
                      bias=False,
                      dilation=1),
            build_norm_layer(inp * expand_ratio, normalize)[1],
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            build_norm_layer(oup, normalize)[1],
        )

    def forward(self, x):
        return self.conv(x)

class ResNet_qavgg(nn.Module):
    """
        layer0 <-> Conv1, ..., layer4 <-> Conv5

        You can configure output layer and its related strides.
    """

    def __init__(self,
                 block,
                 layers,
                 out_layers,
                 out_strides,
                 style='pytorch',
                 frozen_layers=[],
                 layer_deform=[0, 0, 0, 0, 0],
                 normalize={'type': 'freeze_bn'},
                 multiplier=1,
                 checkpoint=False,
                 initializer=None,
                 fa=None,
                 nnie=False,
                 deep_stem=True,
                 input_channel=3,
                 drop_path_rate=0,
                 avg_down=False,
                 down_sample='pool',
                 # width=[64, 128, 256, 512, 512]):
                 width=[64, 64, 128, 256, 512]):
        r"""
        Arguments:
            - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
            - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
            - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
            - style (:obj:`str`): ResNet style (``caffe`` or ``pytorch``), default is ``pytorch`` style.
            - layer_deform (:obj:`list` of (:obj:`str` or None): ``DCN`` setting for each layer. See note below
            - normalize (:obj:`dict`): Config of Normalization Layer (see Configuration#Normalization).
            - checkpoint (:obj:`list` or :obj:`bool`): segments to checkpoint for each layer.
              ``False`` or ``0`` for no checkpoint.
              For more details, refer to `Checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_
            - initializer (:obj:`dict`): Config of initilizaiton
            - fa (:obj:`dict`): Configurations of `FactorizedAttentionBlock <https://arxiv.org/pdf/1812.01243.pdf>`_.


        .. note::

            We support two style ResNet implementation: ``pytorch`` & ``caffe`` style ResNet. The differences between
            them are as follows:

                1. Architecture: ``pytorch`` style ResNet strides in conv3x3 while ``caffe`` style ResNet
                   strides in conv1x1.
                2. Data preprocess: they use different ``pixel_mean`` & ``pixel_std`` in data preprocessing.
                3. Thus, their pretrained weights are different.
                4. ``caffe`` style ResNet support ``caffe_freeze_bn`` only, since the ``running_mean`` and
                   ``running_var`` are absorbed in ``weight`` and ``bias`` in the pretrained checkpoint.

            Here are the configurations needed to be modified when migrating from pytorch style to caffe style ResNet

            .. code-block:: yaml

                dataset:
                    # ...
                    preprocess_style: caffe
                    pixel_mean: [102.9801, 115.9465, 122.7717]  # caffe-style pretrained statistics
                    pixel_std: [1, 1, 1]

                saver:
                    # ...
                    pretrain_model: /mnt/lustre/share/DSK/model_zoo/pytorch/imagenet/R-50-caffe-style.pkl

                net:
                    backbone:
                        # ...
                        normalize:
                            type: caffe_freeze_bn
                        style: caffe
                    # ...
                # ...

        .. seealso::

            Checkpoint purposes on saving GPU memory. It is mplemented by
            rerunning a forward-pass segment for each checkpointed segment during backward.
            Some modules need to adjust their paramters because of checkpointing, such as BN.
            You need change BN momentum to :math:`momentum_{new} = 1 - \sqrt{1 - momentum_{old}}`
            if checkpoint enabled.
            For mote details, refer to `checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_


        .. note::

            Elements of layer_deform support three types: **False**, **last**, **all**, **int**

            * last: only last conv3x3 use Deformable Conv for this layer
            * all: all conv3x3 use Deformable Conv for this layer
            * False: disable Deformable Conv for this layer
            * int(n): last n conv3x3 use Deformable Conv for this layer

            .. code-block:: python

                layer_deform = [False, False, 'last', 1, 'all']

        """

        def check_range(x):
            if x:             # Add conditional operation to avoid error when no frozen layer is provided
                assert min(x) >= 0 and max(x) <= 4, x

        check_range(frozen_layers)
        check_range(out_layers)
        assert len(out_layers) == len(out_strides)

        if style == 'caffe':
            stride_in_1x1 = True
            # assert normalize['type'] == 'caffe_freeze_bn', "Caffe style pretrained only supports 'caffe_freeze_bn'"
        else:
            stride_in_1x1 = False
        self.multiplier = float(multiplier)
        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        layer_out_planes = [width[0] * self.multiplier] + [i * block.expansion * self.multiplier for i in width[1:]]
        layer_in_planes = list(map(int, [width[0] * self.multiplier] + [i * self.multiplier for i in width[1:]]))
        layer_out_planes = list(map(int, layer_out_planes))
        self.out_planes = [layer_out_planes[i] for i in out_layers]
        self.segments = self.get_segments(checkpoint)

        self.inplanes = layer_out_planes[0]
        self.norm1_name, norm1 = build_norm_layer(layer_out_planes[0], normalize, 1)
        self.block_nums = sum(layers)
        self.drop_path_rate = drop_path_rate
        self.avg_down = avg_down

        super(ResNet_qavgg, self).__init__()
        # if not deep_stem:
        #     conv1 = nn.Conv2d(input_channel, layer_out_planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        # else:
        #     inner_planes = layer_out_planes[0]
        #     conv1 = nn.Sequential(
        #         nn.Conv2d(input_channel, inner_planes, kernel_size=3, stride=2, padding=1, bias=False),
        #         build_norm_layer(inner_planes, normalize)[1],
        #         nn.ReLU(inplace=True)
        #
        #     )
        # self.add_module(self.norm1_name, norm1)
        # self.qablock0 = QARepVGGBlock(layer_out_planes[0], layer_out_planes[0])
        # relu = nn.ReLU(inplace=True)
        # if down_sample == 'pool':
        #     maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #     if nnie:
        #         maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        #     self.layer0 = nn.Sequential(conv1, self.qablock0,maxpool)
        # else:
        #     conv2 = nn.Sequential(
        #         nn.Conv2d(layer_out_planes[0], layer_out_planes[0], kernel_size=3, stride=2, padding=1, bias=True),
        #         build_norm_layer(layer_out_planes[0], normalize)[1],
        #         nn.ReLU(inplace=True))
        #     self.layer0 = nn.Sequential(conv1, norm1, relu, conv2)

        self.qablock0 = QARepVGGBlock(input_channel, 32,stride=2)
        self.qablock1 = QARepVGGBlock(32, 32)
        self.qablock2 = QARepVGGBlock(32, 64,stride=2)
        self.qablock3 = QARepVGGBlock(64, 64)
        self.qablock4 = QARepVGGBlock(64, 128)
        # self.qablock6 = QARepVGGBlock(64, 89)

        # self.qablock = QARepVGGBlock(input_channel, 16)
        # self.qablock1 = QARepVGGBlock(16, 16,stride=2)
        # self.qablock3 = QARepVGGBlock(16, 32)
        # self.qablock4 = QARepVGGBlock(32, 32,stride=2)
        # self.qablock5 = QARepVGGBlock(32, 64)
        # self.qablock6 = QARepVGGBlock(64, 64)

        # self.qablock = QARepVGGBlock(input_channel, 16)
        # self.qablock1 = QARepVGGBlock(16, 32,stride=2)
        # self.qablock3 = QARepVGGBlock(32, 64,stride=2)
        # self.qablock4 = QARepVGGBlock(64, 128)
        # self.qablock5 = QARepVGGBlock(128, 89)
        # self.qablock6 = QARepVGGBlock(64, 64)

        self.layer0 = nn.Sequential(self.qablock0)
        self.layer1 = nn.Sequential(self.qablock1)
        self.layer2 = nn.Sequential(self.qablock2)
        self.layer3 = nn.Sequential(self.qablock3,self.qablock4)
        # self.layer3 = nn.Sequential(self.pool,self.hidden )

        # self.layer1 = self._make_layer(block, layer_in_planes[1], layers[0], stride=1,
        #                                normalize=normalize,
        #                                block_id=0)
        # self.layer2 = self._make_layer(block, layer_in_planes[2], layers[1], stride=2,
        #                                normalize=normalize,
        #                                stride_in_1x1=stride_in_1x1,
        #                                block_id=layers[0])
        #
        # self.layer3 = self._make_layer(block, layer_in_planes[3], layers[2], stride=2,
        #                                normalize=normalize,
        #                                stride_in_1x1=stride_in_1x1,
        #                                block_id=sum(layers[0:2]))
        if 4 in self.out_layers:
            layer4_stride = out_strides[-1] // 16
            self.layer4 = self._make_layer(block, layer_in_planes[4], layers[3],
                                           stride=layer4_stride,
                                           dilation=2 // layer4_stride,
                                           normalize=normalize,
                                           stride_in_1x1=stride_in_1x1,
                                           block_id=sum(layers[0:3]))
        else:
            self.layer4 = None

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params
        # to optimizer
        self.freeze_layer()




    def get_segments(self, checkpoint):
        if isinstance(checkpoint, Iterable):
            segments = [int(x) for x in checkpoint]
        else:
            segments = [int(checkpoint)] * 5
        return segments

    def get_fa_layers(self, num_channels, fa_cfg):
        """Build fa blocks in backbone"""
        fa_modules = [None] * 5
        return fa_modules

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    normalize={'type': 'solo_bn'},
                    stride_in_1x1=False,
                    block_id=0):
        block_types = [block] * blocks

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    build_norm_layer(planes * block.expansion, normalize)[1],
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    build_norm_layer(planes * block.expansion, normalize)[1]
                )
        block_drop_path_rate = self.drop_path_rate * block_id / (self.block_nums - 1.)
        block_id += 1
        block_drop_path = DropPath(block_drop_path_rate) if block_drop_path_rate > 0 else None
        layers = []
        layers.append(block_types[0](self.inplanes,
                                     planes,
                                     stride,
                                     dilation,
                                     downsample,
                                     normalize,
                                     stride_in_1x1,
                                     block_drop_path))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            block_drop_path_rate = self.drop_path_rate * block_id / (self.block_nums - 1.)
            block_id += 1
            block_drop_path = DropPath(block_drop_path_rate) if block_drop_path_rate > 0 else None
            layers.append(block_types[i](self.inplanes,
                                         planes,
                                         stride=1,
                                         dilation=dilation,
                                         downsample=None,
                                         normalize=normalize,
                                         stride_in_1x1=stride_in_1x1,
                                         drop_path=block_drop_path
                                         ))
        return nn.Sequential(*layers)

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
        """

        Arguments:
            - input (:obj:`dict`): output of
              :class:`~root.datasets.base_dataset.BaseDataset`

        Returns:
            - out (:obj:`dict`):

        Output example::

            {
                'features': [], # list of tenosr
                'strides': []   # list of int
            }
        """
        if deploy:
            self.qablock0.prepare_for_deploy()
            # self.qablock0.prepare_for_deploy()
            self.qablock1.prepare_for_deploy()
            # self.qablock2.prepare_for_deploy()
            self.qablock2.prepare_for_deploy()
            self.qablock3.prepare_for_deploy()
            self.qablock4.prepare_for_deploy()
            # self.qablock6.prepare_for_deploy()

            # self.qablock7.prepare_for_deploy()
            # self.qablock8.prepare_for_deploy()

        x = input['image']
        # x = input
        outs = []
        for layer_idx in range(0, 5):
            layer = getattr(self, f'layer{layer_idx}', None)
            if layer is not None:  # layer4 is None for C4 backbone
                # Use checkpoint for learnable layer
                if self.segments[layer_idx] > 0 and layer_idx not in self.frozen_layers:
                    x = self.checkpoint_fwd(layer, x, self.segments[layer_idx])
                else:
                    x = layer(x)
                outs.append(x)
        # outs[-1] = self.ca(outs[-1]) * outs[-1]
        # outs[-1] = self.sa(outs[-1]) * outs[-1]
        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        layers = [
            self.layer0,
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self


def resnet18_qavgg(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    block_type = kwargs.pop("block_type", "res")
    Block = BasicBlock


    model = ResNet_qavgg(Block, [1, 1, 1], **kwargs)
    return model


