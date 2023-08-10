import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from SeModule import SELayer
from model.DenseUnet import DenseUnet

import math

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        #out += residual
        out = self.relu(out)

        return out


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

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

class Recurrent_block(nn.Module):
    def __init__(self,ch_out, t=2, kernel_size=3, padding=1):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=kernel_size,stride=1,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class NestedUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = False

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block(img_ch, nb_filter[0])
        self.conv1_0 = conv_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = conv_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = conv_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = conv_block(nb_filter[3], nb_filter[4])

        self.conv0_1 = LS_block(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = LS_block(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = LS_block(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = LS_block(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = LS_block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = LS_block(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = LS_block(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = LS_block(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = LS_block(nb_filter[1]*3+nb_filter[2], nb_filter[1])
        self.conv0_4 = LS_block(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class LUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    
        self.conv0_0 = LS_block(img_ch, nb_filter[0])
        self.conv1_0 = LS_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = LS_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = LS_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = conv_block(nb_filter[3], nb_filter[4])

        self.up3 = up_conv(nb_filter[4], nb_filter[3])
        self.up2 = up_conv(nb_filter[3], nb_filter[2])
        self.up1 = up_conv(nb_filter[2], nb_filter[1])
        self.up0 = up_conv(nb_filter[1], nb_filter[0])

        # self.bottom = PS_module(nb_filter[3], nb_filter[3])

        self.conv3_1 = LS_block(nb_filter[3]+nb_filter[3], nb_filter[3])
        self.conv2_2 = LS_block(nb_filter[2]+nb_filter[2], nb_filter[2])
        self.conv1_3 = LS_block(nb_filter[1]+nb_filter[1], nb_filter[1])
        self.conv0_4 = LS_block(nb_filter[0]+nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # x3_1 = self.bottom(x3_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up0(x1_3)], 1))

        output = self.final(x0_4)
        return output


class PUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    
        self.conv0_0 = conv_block(img_ch, nb_filter[0])
        self.conv1_0 = conv_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = conv_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = conv_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = conv_block(nb_filter[3], nb_filter[4])

        self.up3 = up_conv(nb_filter[4], nb_filter[3])
        self.up2 = up_conv(nb_filter[3], nb_filter[2])
        self.up1 = up_conv(nb_filter[2], nb_filter[1])
        self.up0 = up_conv(nb_filter[1], nb_filter[0])

        self.bottom = PS_module(nb_filter[4], nb_filter[4])

        self.conv3_1 = conv_block(nb_filter[3]+nb_filter[3], nb_filter[3])
        self.conv2_2 = conv_block(nb_filter[2]+nb_filter[2], nb_filter[2])
        self.conv1_3 = conv_block(nb_filter[1]+nb_filter[1], nb_filter[1])
        self.conv0_4 = conv_block(nb_filter[0]+nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x4_1 = self.bottom(x4_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3(x4_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up0(x1_3)], 1))

        output = self.final(x0_4)
        return output

class PLUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    
        self.conv0_0 = LS_block(img_ch, nb_filter[0])
        self.conv1_0 = LS_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = LS_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = LS_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = LS_block(nb_filter[3], nb_filter[4])

        self.up3 = up_conv(nb_filter[4], nb_filter[3])
        self.up2 = up_conv(nb_filter[3], nb_filter[2])
        self.up1 = up_conv(nb_filter[2], nb_filter[1])
        self.up0 = up_conv(nb_filter[1], nb_filter[0])

        self.bottom = PS_module(nb_filter[4], nb_filter[4])

        self.conv3_1 = LS_block(nb_filter[3]+nb_filter[3], nb_filter[3])
        self.conv2_2 = LS_block(nb_filter[2]+nb_filter[2], nb_filter[2])
        self.conv1_3 = LS_block(nb_filter[1]+nb_filter[1], nb_filter[1])
        self.conv0_4 = LS_block(nb_filter[0]+nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x4_1 = self.bottom(x4_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3(x4_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up0(x1_3)], 1))

        output = self.final(x0_4)
        return output


class PLUNet32(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    
        self.conv0_0 = conv_block(img_ch, nb_filter[0])
        self.conv1_0 = conv_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = conv_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = conv_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = conv_block(nb_filter[3], nb_filter[4])

        self.up3 = up_conv(nb_filter[4], nb_filter[3])
        self.up2 = up_conv(nb_filter[3], nb_filter[2])
        self.up1 = up_conv(nb_filter[2], nb_filter[1])
        self.up0 = up_conv(nb_filter[1], nb_filter[0])


        self.conv3_1 = Bottle2neck(nb_filter[3]+nb_filter[3], nb_filter[3])
        self.conv2_2 = Bottle2neck(nb_filter[2]+nb_filter[2], nb_filter[2])
        self.conv1_3 = Bottle2neck(nb_filter[1]+nb_filter[1], nb_filter[1])
        self.conv0_4 = Bottle2neck(nb_filter[0]+nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))


        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up0(x1_3)], 1))

        output = self.final(x0_4)
        return output

class PLUNet512(nn.Module):
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

        self.bottom = PS_module(nb_filter[3], nb_filter[3])

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
