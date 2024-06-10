import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, NonBottleneck1D
from resnet import ResNet18 as ResNet18_
from resnet import ResNet34 as ResNet34_
from resnet import ResNet50 as ResNet50_
from resnet import ResNet101 as ResNet101_
from resnet import ResNet152 as ResNet152_
from low_high import AFNB, AFNB2
from decoder import Decoder as Decoder2
from decoder import SegFormerHead as MLPDecoder
from decoder import SFAFDecoder


class RGBTNet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=9,
                 encoder_1='resnet18',
                 encoder_2='mit_b1',
                 encoder_block='BasicBlock',
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation=nn.ReLU,
                 pretrained=True,
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):
        super(RGBTNet, self).__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.register_buffer('probability', torch.eye(num_classes, requires_grad=False))
        # encoder

        if encoder_1 == 'resnet18':
            self.encoder_rgb = ResNet18_(pretrained_on_imagenet=pretrained_on_imagenet, pretrained_dir=pretrained_dir)
            self.encoder_th = ResNet18(pretrained_on_imagenet=pretrained_on_imagenet, pretrained_dir=pretrained_dir)
        elif encoder_1 == 'resnet34':
            self.encoder_rgb = ResNet34_(pretrained_on_imagenet=pretrained_on_imagenet, pretrained_dir=pretrained_dir)
            self.encoder_th = ResNet34(pretrained_on_imagenet=pretrained_on_imagenet, pretrained_dir=pretrained_dir)
        elif encoder_1 == 'resnet50':
            self.encoder_rgb = ResNet50_(pretrained_on_imagenet=pretrained_on_imagenet)
            self.encoder_th = ResNet50(pretrained_on_imagenet=pretrained_on_imagenet)#, input_channels=1)
        elif encoder_1 == 'resnet101':
            self.encoder_rgb = ResNet101_(pretrained_on_imagenet=pretrained_on_imagenet)
            self.encoder_th = ResNet101(pretrained_on_imagenet=pretrained_on_imagenet)#, input_channels=1)
        elif encoder_1 == 'resnet152':
            self.encoder_rgb = ResNet152_(pretrained_on_imagenet=pretrained_on_imagenet)
            self.encoder_th = ResNet152(pretrained_on_imagenet=pretrained_on_imagenet)#, input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_1))

        self.layers_channel = [self.encoder_rgb.down_4_channels_out, self.encoder_rgb.down_8_channels_out,
                               self.encoder_rgb.down_16_channels_out, self.encoder_rgb.down_32_channels_out]

        self.skip_layer0 = AFNB2(low_in_channels=self.encoder_rgb.down_2_channels_out,
                                high_in_channels=self.layers_channel[0],
                                out_channels=self.layers_channel[0],
                                key_channels=self.layers_channel[0] // 4, value_channels=self.layers_channel[0] // 4,
                                dropout=0.05, psp_size=[1, 3, 6, 8, 12])
        
        self.skip_layer1 = AFNB2(low_in_channels=self.layers_channel[0],
                                high_in_channels=self.layers_channel[1],
                                out_channels=self.layers_channel[1],
                                key_channels=self.layers_channel[0] // 4, value_channels=self.layers_channel[0] // 4,
                                dropout=0.05, psp_size=[1, 3, 6, 8, 12])
        
        self.skip_layer2 = AFNB2(low_in_channels=self.layers_channel[1],
                                high_in_channels=self.layers_channel[2],
                                out_channels=self.layers_channel[2],
                                key_channels=self.layers_channel[1] // 4, value_channels=self.layers_channel[1] // 4,
                                dropout=0.05, psp_size=[1, 3, 6, 8, 12])
        # self.skip_layer0 = Add()
        # self.skip_layer1 = Add()
        # self.skip_layer2 = Add()
        # self.fuse = Add()

        self.decoder = Decoder2(channels_decoder=self.layers_channel, num_classes=num_classes)
        self.fuse = FusionBlock(in_channel=self.layers_channel[3], dilation_rates=[1, 3, 5, 7])

    def forward(self, images):
        rgb, thermal = images[:, 0:3], images[:, 3:]
        thermal = torch.cat([thermal, thermal, thermal], dim=1)
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        thermal = self.encoder_th.forward_first_conv(thermal)

        rgb_layer1 = F.max_pool2d(rgb, kernel_size=3, stride=2, padding=1)
        thermal_layer1 = F.max_pool2d(thermal, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb_layer1 = self.encoder_rgb.forward_layer1(rgb_layer1)
        thermal_layer1 = self.encoder_th.forward_layer1(thermal_layer1)
        skip0 = self.skip_layer0([rgb, thermal], rgb_layer1, thermal_layer1)

        # block 2
        rgb_layer2 = self.encoder_rgb.forward_layer2(rgb_layer1)
        thermal_layer2 = self.encoder_rgb.forward_layer2(thermal_layer1)
        skip1 = self.skip_layer1([rgb_layer1, thermal_layer1], rgb_layer2, thermal_layer2)

        # block 3
        rgb_layer3 = self.encoder_rgb.forward_layer3(rgb_layer2)
        thermal_layer3 = self.encoder_rgb.forward_layer3(thermal_layer2)
        skip2 = self.skip_layer2([rgb_layer2, thermal_layer2], rgb_layer3, thermal_layer3)

        # block 4
        rgb_layer4 = self.encoder_rgb.forward_layer4(rgb_layer3)
        thermal_layer4 = self.encoder_rgb.forward_layer4(thermal_layer3)
        fusion = self.fuse(rgb_layer4, thermal_layer4)

        # decoder
        out = self.decoder([fusion, skip2, skip1, skip0])
        return out


class Add(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        return x1 + x2


class FusionBlock(nn.Module):
    def __init__(self, in_channel, dilation_rates=[1, 3, 5]):
        super(FusionBlock, self).__init__()

        self.dilation_conv = nn.ModuleList()
        for index in range(len(dilation_rates)):
            self.dilation_conv.append(
                #nn.Conv2d(2 * in_channel, in_channel, 3, 1, padding=dilation_rates[index], dilation=dilation_rates[index]))
            
                nn.Sequential(
                    nn.Conv2d(2 * in_channel, 2 * in_channel, 3, 1, padding=dilation_rates[index],
                              dilation=dilation_rates[index], groups=2 * in_channel),
                    nn.Conv2d(2 * in_channel, in_channel, 1, 1)
                )
            )
        self.conv1x1 = nn.Conv2d(2 * in_channel, in_channel, 1, 1)
        self.merge = nn.Sequential(
            nn.Conv2d(2 * in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(num_features=in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, thermal):
        merge = torch.cat([rgb, thermal], dim=1)
        # merge = self.fuse(merge)

        out = 0.
        for dilation_layer in self.dilation_conv:
            out += dilation_layer(merge)
        fuse = self.conv1x1(merge)
        out = torch.cat([out, fuse], dim=1)
        out = self.merge(out)
        return out


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)

