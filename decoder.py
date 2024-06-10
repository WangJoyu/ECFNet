import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.distributions import normal


class Decoder(nn.Module):
    def __init__(self,
                 channels_decoder,
                 upsampling_mode='bilinear',
                 num_classes=9,
                 branch=3):
        super(Decoder, self).__init__()

        self.decoder_module_3 = nn.ModuleList([ChannelSpatialAttention(channels_decoder[3], channels_decoder[2])])

        self.decoder_module_2 = nn.ModuleList([ChannelSpatialAttention(channels_decoder[2], channels_decoder[1]),
                                               ChannelSpatialAttention(channels_decoder[2], channels_decoder[1])])

        self.decoder_module_1 = nn.ModuleList([ChannelSpatialAttention(channels_decoder[1], channels_decoder[0]),
                                               ChannelSpatialAttention(channels_decoder[1], channels_decoder[0]),
                                               ChannelSpatialAttention(channels_decoder[1], channels_decoder[0])])
        
        self.conv_out = nn.ModuleList([nn.Conv2d(channels_decoder[0], num_classes, kernel_size=3, padding=1),
                                        nn.Conv2d(channels_decoder[0], num_classes, kernel_size=3, padding=1),
                                        nn.Conv2d(channels_decoder[0], num_classes, kernel_size=3, padding=1)])

        self.upsample = nn.Upsample(scale_factor=4, mode=upsampling_mode, align_corners=False)
        self.branch = branch
        self.alpha = nn.Parameter(torch.ones(self.branch, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def distance(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1).unsqueeze(-2)
        x = (x - self.base_coord) ** 2  # 
        x = torch.sum(x * self.beta, dim=-1)
        #x = x ** 0.5
        x = x.permute(0, 2, 1).reshape(b, -1, h, w)
        return x
    
    def forward(self, enc_outs):
        
        enc_skip3, enc_skip2, enc_skip1, enc_skip0 = enc_outs
        outs = []
        # branch 1
        enc_skip2 = self.decoder_module_3[0](enc_skip3, enc_skip2)
        enc_skip1 = self.decoder_module_2[0](enc_skip2, enc_skip1)
        enc_skip0 = self.decoder_module_1[0](enc_skip1, enc_skip0)
        out = self.conv_out[0](enc_skip0)
        outs = outs + [self.upsample(out)]
        
        # branch 2
        enc_skip1 = self.decoder_module_2[1](enc_skip2, enc_skip1)
        enc_skip0 = self.decoder_module_1[1](enc_skip1, enc_skip0)
        out = self.conv_out[1](enc_skip0)
        outs = outs + [self.upsample(out)]
        
        # branch 3
        enc_skip0 = self.decoder_module_1[2](enc_skip1, enc_skip0)
        out = self.conv_out[2](enc_skip0)
        outs = outs + [self.upsample(out)]
        
        # output
        alpha_soft = F.softmax(self.alpha, dim=0)
        out = 0.
        for ind in range(self.branch):
            out += outs[ind].detach() * alpha_soft[ind]

        outs = outs + [out]
        if self.training:
            return outs
        return out


class BLV(nn.Module):
    def __init__(self,
                 cls_num_list=[442502553, 21803212, 6645964, 2057011, 3101491, 2624102, 726220, 841728, 1378099],
                 sigma=6):
        super(BLV, self).__init__()
        cls_list = torch.tensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        frequency_list = frequency_list.max() - frequency_list
        self.frequency_list = frequency_list / frequency_list.max()
        self.sampler = normal.Normal(0, sigma)

    def forward(self, pred):
        viariation = self.sampler.sample(pred.shape).to(pred.device)
        pred = pred + (viariation.abs().permute(0, 2, 3, 1) * self.frequency_list.to(pred.device)).permute(0, 3, 1, 2)
        return pred
        
        
class ChannelSpatialAttention(nn.Module):
    def __init__(self, high_channel, low_channel,):
        super(ChannelSpatialAttention, self).__init__()

        self.spatial_attention = SpatialAttention(in_channel=low_channel, dec_channel=low_channel//8)
        self.channel_attention = SqueezeAndExcitation(channel=high_channel)
        self.upsample_smooth = nn.Sequential(nn.Upsample(scale_factor=2),
                                             nn.AvgPool2d(3, 1, 1))
        self.conv = nn.Conv2d(low_channel, high_channel, 3, 1, 1)
        self.conv_high = nn.Sequential(nn.Conv2d(high_channel, high_channel, 3, 1, 1),
                                       nn.BatchNorm2d(high_channel),
                                       nn.ReLU(True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(2*high_channel, low_channel, 1),
                                       nn.BatchNorm2d(low_channel),
                                       nn.ReLU(True))

    def forward(self, high_level, low_level):
        low_level = self.spatial_attention(low_level)
        low_level = self.conv(low_level)

        high_level = self.conv_high(high_level)
        high_level = self.channel_attention(high_level)
        high_level = self.upsample_smooth(high_level)
        out = torch.cat([low_level, high_level], dim=1)
        out = self.conv_fuse(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channel, dec_channel):
        super(SpatialAttention, self).__init__()

        # self.conv1 = nn.Sequential(nn.Conv2d(in_channel, dec_channel, 1, 1),
        #                            nn.BatchNorm2d(dec_channel),
        #                            nn.ReLU(True),
        #                            nn.Conv2d(dec_channel, 1, 1),
        #                            nn.BatchNorm2d(1),
        #                            nn.Sigmoid())
        self.max_pooling = nn.MaxPool1d(in_channel, in_channel)
        self.avg_pooling = nn.AvgPool1d(in_channel, in_channel)
        self.fc = nn.Sequential(nn.Conv2d(2, 1, 1),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        weights = x.reshape(B, C, -1).permute(0, 2, 1)
        max_pooling = self.max_pooling(weights)
        avg_pooling = self.avg_pooling(weights)
        weights = torch.cat([max_pooling, avg_pooling], dim=-1).permute(0, 2, 1).reshape(B, 2, H, W)
        weights = self.fc(weights)
        # weights = self.conv1(x)
        x = x * weights
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, k_size=3, activation=nn.ReLU(inplace=True)):
        super(ChannelAttention, self).__init__()
        if channel == 64 or channel == 128:
            k_size = 3
        else:
            k_size = 5
        self.fuse = nn.Linear(3, 1)
        self.proj = nn.Sequential(
            nn.Conv1d(1, 1, k_size, 1, k_size//2),
            activation,
            nn.Conv1d(1, 1, k_size, 1, k_size//2),
            nn.Sigmoid()
        )
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        B, C, H, W = x.shape
        weighting_mean = self.mean(x).squeeze(-1)
        weighting_max = self.max(x).squeeze(-1)
        weighting_var = torch.var(x.reshape(B, C, -1), dim=-1, keepdim=True)
        weighting = torch.cat([weighting_mean, weighting_var, weighting_max], dim=-1)  # B C 3
        weighting = self.fuse(weighting).permute(0, 2, 1)  # B C 1
        weighting = self.proj(weighting).permute(0, 2, 1).unsqueeze(-1)  # B C 1 1
        x = x * weighting
        return x
        
        
class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        weighting = self.pooling(x)
        weighting = self.fc(weighting)
        x = x * weighting
        return x

def normalization(x):
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1)
    x = x - torch.min(x, dim=-1, keepdim=True)[0]
    #x = x / torch.max(x, dim=-1, keepdim=True)[0]
    #x = 100 * x.reshape(B, C, H, W)
    x = x.reshape(B, C, H, W)
    return x
    

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        # self.linear_fuse = ConvModule(
        #     in_channels=embedding_dim*4,
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='SyncBN', requires_grad=True)
        # )
        self.linear_fuse = nn.Sequential(nn.Conv2d(embedding_dim*4, embedding_dim, 1),
                                          nn.BatchNorm2d(embedding_dim),
                                          nn.ReLU(inplace=True))

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):

        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.upsample(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
   
class SFAFDecoder(nn.Module):
    def __init__(self, filters, num_classes=9):
        super(SFAFDecoder, self).__init__()
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
    
    def forward(self, x):
        e0, e1, e2, e3, e4 = x
        d4 = self.decoder4(e4)+e3
        d3 = self.decoder3(d4)+e2
        d2 = self.decoder2(d3)+e1
        d1 = self.decoder1(d2)+e0
        fuse = self.finaldeconv1(d1)
        fuse = self.finalrelu1(fuse)
        fuse = self.finalconv2(fuse)
        fuse = self.finalrelu2(fuse)
        fuse = self.finalconv3(fuse)

        return fuse