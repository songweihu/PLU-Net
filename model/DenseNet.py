import torch
import torch.nn as nn
import torch.nn.functional as F
from SeModule import SELayer

from queue import Queue
from torch.autograd import Variable

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features

class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.se = SELayer(inter_channels2, 4)
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)

        self.conv1x1 = nn.Conv2d(in_channels + inter_channels2 * 5, inter_channels2, 1, 1)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        x = self.conv1x1(x)
        #x = self.se(x)
        return x

class PS_module(nn.Module):
    def __init__(self, in_channels, depth):
        super(PS_module,self).__init__()
        
        self.conv6 = nn.Conv2d(in_channels, depth//4, kernel_size=1)
        self.atrous_block1 = nn.Conv2d(depth//4, depth//4, 1, 1,)
        self.atrous_block6 = nn.Conv2d(depth//4, depth//4, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(depth//4, depth//4, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(depth//4, depth//4, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth, depth, 1, 1)
        self.se = SELayer(depth, 4)
    def forward(self, x):
        
        x2 = self.conv6(x)
        atrous_block1 = self.atrous_block1(x2)
        atrous_block6 = self.atrous_block6(x2)
        atrous_block12 = self.atrous_block12(x2)
        atrous_block18 = self.atrous_block18(x2)
 
        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        out = self.se(net)
        return out

class LS_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(LS_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out//2),
            nn.ReLU(inplace=True),
        )
        self.dconv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, kernel_size=3,stride=1,padding=3,dilation=3,bias=False),
            nn.BatchNorm2d(ch_out//2),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = nn.Conv2d(ch_out, ch_out, 1, 1, bias=False)
        self.se = SELayer(ch_out, 4)

    def forward(self,x):
        conv = self.conv(x)
        dconv = self.dconv(x)
        # cat = self.se(torch.cat((conv, dconv), 1))
        cat = self.se(self.conv1x1(torch.cat((conv, dconv),1)))
        return cat


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class DenseNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super().__init__()

        nb_filter = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    
        self.conv0_0 = LS_block(img_ch, nb_filter[0])
        self.conv1_0 = LS_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = LS_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = LS_block(nb_filter[2], nb_filter[3])
        # self.conv4_0 = LS_block(nb_filter[3], nb_filter[4])

        # self.up3 = up_conv(nb_filter[4], nb_filter[3])
        self.up2 = up_conv(nb_filter[3], nb_filter[2])
        self.up1 = up_conv(nb_filter[2], nb_filter[1])
        self.up0 = up_conv(nb_filter[1], nb_filter[0])

        self.bottom = _DenseASPPBlock(nb_filter[3], nb_filter[3], nb_filter[3])

        # self.conv3_1 = LS_block(nb_filter[3]+nb_filter[3], nb_filter[3])
        self.conv2_2 = LS_block(nb_filter[2]+nb_filter[2], nb_filter[2])
        self.conv1_3 = LS_block(nb_filter[1]+nb_filter[1], nb_filter[1])
        self.conv0_4 = LS_block(nb_filter[0]+nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        # x4_0 = self.conv4_0(self.pool(x3_0))

        x4_1 = self.bottom(x3_0)

        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up3(x4_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2(x4_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up0(x1_3)], 1))

        output = self.final(x0_4)
        return output