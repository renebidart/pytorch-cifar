# From: https://github.com/rwightman/pytorch-image-models
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn import functional as F

def _get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class SEModule(nn.Module):
    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        #x_se = self.avg_pool(x)
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.fc1(x_se)
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1

class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg'):
        super(SelectAdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        if pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                assert False, 'Invalid pool type: %s' % pool_type
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'output_size=' + str(self.output_size) \
               + ', pool_type=' + self.pool_type + ')'

class BasicBlockLightOLD(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, ws_factor, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False,
                 reduce_first=1, dilation=1, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockLight, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        self.inplanes = inplanes
        self.first_planes = planes // reduce_first
        self.outplanes = planes * self.expansion
        self.planes = planes
        self.ws_factor = ws_factor
#         print('self.first_planes', self.first_planes)
#         print('self.ws_factor', self.ws_factor)

        self.conv1 = nn.Conv2d(
            self.inplanes, int(self.first_planes/self.ws_factor), kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False)
        self.bn1 = norm_layer(self.first_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            self.first_planes, int(self.outplanes/self.ws_factor), kernel_size=3, padding=previous_dilation,
            dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(self.outplanes)
        self.se = SEModule(self.outplanes, planes // 4) if use_se else None
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = out.repeat(1, self.ws_factor, 1, 1)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out.repeat(1, self.ws_factor, 1, 1)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    
class BasicBlockLight(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, ws_factor, downsample_repeat=True, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False,
                 reduce_first=1, dilation=1, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockLight, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        self.inplanes = inplanes
        self.first_planes = planes // reduce_first
        self.outplanes = planes * self.expansion
        self.planes = planes
        self.ws_factor = ws_factor
        self.downsample_repeat = downsample_repeat

        self.conv1 = nn.Conv2d(
            self.inplanes, int(self.first_planes/self.ws_factor), kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False)
        self.bn1 = norm_layer(int(self.first_planes/self.ws_factor))
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            self.first_planes, int(self.outplanes/self.ws_factor), kernel_size=3, padding=previous_dilation,
            dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(int(self.outplanes/self.ws_factor))
        self.se = SEModule(self.outplanes, planes // 4) if use_se else None
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = out.repeat(1, self.ws_factor, 1, 1)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out.repeat(1, self.ws_factor, 1, 1)

        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
            residual = residual.repeat(1, self.ws_factor, 1, 1) if self.downsample_repeat else residual
        out += residual
        out = self.act2(out)
        return out

    
class BottleneckLight(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, ws_factor, downsample_repeat=True, stride=1, downsample=None,
                 cardinality=1, base_width=64, use_se=False,
                 reduce_first=1, dilation=1, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckLight, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        
        self.inplanes = inplanes
        self.first_planes = planes // reduce_first
        self.planes = planes
        self.width = width
        self.outplanes = planes * self.expansion
        self.ws_factor = ws_factor
        self.downsample_repeat = downsample_repeat
        
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)        
        self.act3 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, int(self.first_planes/self.ws_factor), kernel_size=1, bias=False)
        self.bn1 = norm_layer(int(self.first_planes/self.ws_factor))
        self.conv2 = nn.Conv2d(
            self.first_planes, int(self.width/self.ws_factor), kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(int(self.width/self.ws_factor))
        self.conv3 = nn.Conv2d(self.width, int(self.outplanes/self.ws_factor), kernel_size=1, bias=False)
        self.bn3 = norm_layer(int(self.outplanes/self.ws_factor))
        self.se = SEModule(self.outplanes, self.planes // 4) if use_se else None
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = out.repeat(1, self.ws_factor, 1, 1)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out.repeat(1, self.ws_factor, 1, 1)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = out.repeat(1, self.ws_factor, 1, 1)

        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
            residual = residual.repeat(1, self.ws_factor, 1, 1) if self.downsample_repeat else residual
        out += residual
        out = self.act3(out)
        return out


class ResNetLight(nn.Module):
    """ Resnet, but with weight sharing in blocks 2, 3, 4 
    (skip block 1 because few filters, probably won't go well, but just a guess.)
    This is just simple weight sharing or removing filters, nothing fancy
    
    NO idea how to do this properly, this is all really the duplication, so not sure will work.
    Also extra uncertain about bottleneck, because the 3x3 operates on a much smaller dimension.
    No weight sharing in downsample layers either
    !!!!!!!!
    
    ResNet / ResNeXt / SE-ResNeXt / SE-Net
    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering
    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.
    ResNet variants:
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32
      * d - 3 layer deep 3x3 stem, stem_width = 32, average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64, average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64
    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled
    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled
    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    use_se : bool, default False
        Enable Squeeze-Excitation module in blocks
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    stem_width : int, default 64
        Number of channels in stem convolutions
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """
    def __init__(self, block, layers, channel_factor, ws_factor, downsample_repeat, cifar, num_classes=1000, in_chans=3, use_se=False,
                 cardinality=1, base_width=64, stem_width=64, deep_stem=False,
                 block_reduce_first=1, down_kernel_size=1, avg_down=False, dilated=False,
                 norm_layer=nn.BatchNorm2d, drop_rate=0.0, global_pool='avg', drop_connect_rate=None):
        self.cifar = cifar
        self.num_classes = num_classes
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        self.expansion = block.expansion
        self.dilated = dilated
        
        self.ws_factor = ws_factor
        self.channel_factor = channel_factor
        self.downsample_repeat = downsample_repeat

        super(ResNetLight, self).__init__()

        if deep_stem:
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_width, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, self.inplanes, 3, stride=1, padding=1, bias=False)])
        elif self.cifar:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride_3_4 = 1 if self.dilated else 2
        dilation_3 = 2 if self.dilated else 1
        dilation_4 = 4 if self.dilated else 1
        largs = dict(use_se=use_se, reduce_first=block_reduce_first, norm_layer=norm_layer,
                     avg_down=avg_down, down_kernel_size=down_kernel_size)
        self.layer1 = self._make_layer(block, 64, 1, downsample_repeat=1,
                                       blocks=layers[0], stride=1, **largs)
        self.layer2 = self._make_layer(block, int(128/self.channel_factor), self.ws_factor, downsample_repeat,
                                       blocks=layers[1], stride=2, **largs)
        self.layer3 = self._make_layer(block, int(256/self.channel_factor), self.ws_factor, downsample_repeat, 
                                       blocks=layers[2], stride=stride_3_4, dilation=dilation_3, **largs)
        self.layer4 = self._make_layer(block, int(512/self.channel_factor), self.ws_factor, downsample_repeat, 
                                       blocks=layers[3], stride=stride_3_4, dilation=dilation_4, **largs)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_features = int(512/self.channel_factor) * block.expansion
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def _make_layer(self, block, planes, ws_factor, downsample_repeat, blocks, stride=1, dilation=1, reduce_first=1,
                    use_se=False, avg_down=False, down_kernel_size=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        down_kernel_size = 1 if stride == 1 and dilation == 1 else down_kernel_size
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_padding = _get_padding(down_kernel_size, stride)
            downsample_layers = []
            conv_stride = stride
            if avg_down:
                avg_stride = stride if dilation == 1 else 1
                conv_stride = 1
                downsample_layers = [nn.AvgPool2d(avg_stride, avg_stride, ceil_mode=True, count_include_pad=False)]
            if self.downsample_repeat:
                downsample_layers += [
                    nn.Conv2d(self.inplanes, int(planes * block.expansion/self.ws_factor), down_kernel_size,
                              stride=conv_stride, padding=downsample_padding, bias=False),
                    norm_layer(int(planes * block.expansion/self.ws_factor))]
            else:
                downsample_layers += [
                    nn.Conv2d(self.inplanes, planes * block.expansion, down_kernel_size,
                              stride=conv_stride, padding=downsample_padding, bias=False),
                    norm_layer(planes * block.expansion)]
            downsample = nn.Sequential(*downsample_layers)

        first_dilation = 1 if dilation in (1, 2) else 2
        layers = [block(
            self.inplanes, planes, ws_factor, downsample_repeat, stride, downsample,
            cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
            use_se=use_se, dilation=first_dilation, previous_dilation=dilation, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, ws_factor, downsample_repeat,
                cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
                use_se=use_se, dilation=dilation, previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.fc
        if num_classes:
            self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.fc = None

    def forward_features(self, x, pool=True):
#         print('0', x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.cifar else self.maxpool(x)
#         print('0', x.size())
        x = self.layer1(x)
#         print('1', x.size())
        x = self.layer2(x)
#         print('2', x.size())
        x = self.layer3(x)
#         print('3', x.size())
        x = self.layer4(x)
#         print('4', x.size())
        if pool:
            x = self.global_pool(x)
#             print('pool', x.size())
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x