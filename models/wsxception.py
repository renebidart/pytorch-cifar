"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .registry import register_model
# from .helpers import load_pretrained
# from .adaptive_avgmax_pool import select_adaptive_pool2d

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
    
def select_adaptive_pool2d(x, pool_type='avg', output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x
    
    
class LightSeparableConv2d(nn.Conv2d):
    """Weight sharing for both separable spatial conv, and 1x1 conv
    
    WS in kxk seperable conv is repeating the weight. (to size input_channels)
    WS in 1x1 is repeating the output. (to size out_channels)
    
    Probably should change to nn.Module, but its easier to leave liek this to handle paired vars
    """
    def __init__(self, input_channels, out_channels, spatial_nbk, channel_nbk, kernel_size, 
                 stride, padding, dilation=1, bias=False):
        
        super(LightSeparableConv2d, self).__init__(
              in_channels=input_channels, out_channels=out_channels, kernel_size=kernel_size,
              stride=stride, padding=padding, groups=1, dilation=dilation, bias=bias)
        
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.spatial_nbk = spatial_nbk
        self.channel_nbk = channel_nbk
        print(f'LightC2d in_ch: {self.input_channels}, spatial_nbk {self.spatial_nbk}, channel_nbk {self.channel_nbk}')
        self.weight = nn.Parameter(torch.Tensor(spatial_nbk, 1, self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.spatial_nbk))
            self.point_bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.bias = None
            self.point_bias = None
        
        self.point_weight = nn.Parameter(torch.Tensor(self.out_channels, self.channel_nbk, 1, 1))
        nn.init.xavier_uniform_(self.point_weight)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        x = F.conv2d(x,
                    self.weight.repeat(int(self.input_channels/self.spatial_nbk), 1, 1, 1), 
                    self.bias, self.stride,
                    self.padding, self.dilation, groups=self.input_channels)
        
        x = F.conv2d(x,
                    self.point_weight.repeat(1, int(self.input_channels/self.channel_nbk), 1, 1),
                    bias=self.point_bias, stride=1, padding=0, dilation=1, groups=1)
        return x


class LightBlock(nn.Module):
    def __init__(self, in_filters, out_filters, spatial_bf, channel_bf, reps, strides=1, start_with_relu=True, grow_first=True):
        super(LightBlock, self).__init__()
        self.sbf = spatial_bf
        self.cbf = channel_bf

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(LightSeparableConv2d(in_filters, out_filters, int(in_filters/self.sbf), int(in_filters/self.cbf), 
                                            kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(LightSeparableConv2d(filters, filters, int(filters/self.sbf), int(filters/self.cbf), 
                                            kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(LightSeparableConv2d(in_filters, out_filters, int(in_filters/self.sbf), int(in_filters/self.cbf), 
                                            kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class XceptionLight(nn.Module):
    """ Xception with weight sharing added in blocks 4-12, 
    excluding first few and last few to simplify problem
    
    small change 728 -> 736 to divide evenly
    No even way to reduce total channels like in weight sharing, input/outpus layers must also have fewer channels. 
    Choose to keep last input layer same dim, but reduce dim on last output
    
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, spatial_bf, channel_bf, total_f, cifar, num_classes=1000, in_chans=3, drop_rate=0.1, global_pool='avg', 
                drop_connect_rate=.1):
        """ CIFAR version does less downsampling intitially, does a downsample in the middle of the blocks, drops 2 blocks, 
            changes dim from 736 to 512
            
            wieght sharing is only in the 9/12 (or 5/7 if cifar) blocks that use the full dim, the first 2 or 3 are skipped
        """
        super(XceptionLight, self).__init__()
        self.spatial_bf = spatial_bf
        self.channel_bf = channel_bf
        self.total_f = total_f
        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.cifar = cifar

        if self.cifar:
            self.dim = 512
            self.num_features = 512
            self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)

        else:
            self.dim = 736 # originally 728
            self.num_features = 2048
            self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)  
            self.conv2 = nn.Conv2d(32, 64, 3, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here
        
        if self.cifar:
            self.block1 = Block(64, 128, 2, strides=1, start_with_relu=False, grow_first=True)
            self.block2 = None #Block(128, 256, 2, strides=1, start_with_relu=True, grow_first=True)
            self.block3 = Block(128, int(self.dim/self.total_f), 2, strides=2, start_with_relu=True, grow_first=True)
        else:
            self.block1 = Block(64, 128, 2, strides=2, start_with_relu=False, grow_first=True)
            self.block2 = Block(128, 256, 2, strides=2, start_with_relu=True, grow_first=True)
            self.block3 = Block(256, int(self.dim/self.total_f), 2, strides=2, start_with_relu=True, grow_first=True)

        self.block4 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                 reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block5 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                 reps=3, strides=1, start_with_relu=True, grow_first=True)
        if self.cifar:
            self.block6 = None
            self.block7 = None
            self.block8 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                         reps=3, strides=2, start_with_relu=True, grow_first=True)
            self.block9 = None
            self.block10 = None
        else:
            self.block6 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                     reps=3, strides=1, start_with_relu=True, grow_first=True)
            self.block7 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                         reps=3, strides=1, start_with_relu=True, grow_first=True)
            self.block8 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                     reps=3, strides=1, start_with_relu=True, grow_first=True)
            self.block9 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                     reps=3, strides=1, start_with_relu=True, grow_first=True)
            self.block10 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                      reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block11 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                  reps=3, strides=1, start_with_relu=True, grow_first=True)

        self.block12 = LightBlock(int(self.dim/self.total_f), int(1024/self.total_f), self.spatial_bf, self.channel_bf, 
                                  reps=2, strides=2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(int(1024/self.total_f), 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, self.num_features, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(self.num_features)

        self.fc = nn.Linear(self.num_features, num_classes)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = global_pool
        del self.fc
        if num_classes:
            self.fc = nn.Linear(self.num_features, num_classes)
        else:
            self.fc = None

    def forward_features(self, input, pool=True):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        if not self.cifar:
            x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        if self.cifar:
            x = self.block8(x)
        else:
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        if pool:
            x = select_adaptive_pool2d(x, pool_type=self.global_pool)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, input):
        x = self.forward_features(input)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


    
class VSXception(nn.Module):
    def __init__(self, spatial_bf, channel_bf, total_f, num_classes=1000, in_chans=3):
        super(VSXception, self).__init__()
        self.spatial_bf = spatial_bf
        self.channel_bf = channel_bf
        self.total_f = total_f
        self.num_classes = num_classes

        self.dim = 256
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 128, 3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(128)

#         self.block1 = Block(64, int(self.dim/self.total_f), 2, strides=2, start_with_relu=False, grow_first=True)
        self.block1 = LightBlock(128, int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                 reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block2 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                 reps=3, strides=2, start_with_relu=True, grow_first=True)
        self.block3 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
                                 reps=3, strides=2, start_with_relu=True, grow_first=True)
#         self.block4 = LightBlock(int(self.dim/self.total_f), int(self.dim/self.total_f), self.spatial_bf, self.channel_bf, 
#                                   reps=3, strides=2, start_with_relu=True, grow_first=True)
        self.fc = nn.Linear(int(self.dim/self.total_f), num_classes)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
#         x = self.block4(x)
                    
        x = select_adaptive_pool2d(x, pool_type='avg')
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
