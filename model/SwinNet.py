import torch
import torch.nn as nn

class swinconv_block(nn.Module):
    def __init__(self,ch_in,ch_out, size):
        super(swinconv_block,self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        if size == 224:
            self.dilation1 = 1
            self.dilation2 = 4
        elif size == 112:
            self.dilation1 = 2
            self.dilation2 = 6
        elif size == 56:
            self.dilation1 = 3
            self.dilation2 = 8
        elif size == 28:
            self.dilation1 = 4
            self.dilation2 = 10
        elif size == 3:
            self.dilation1 = 2
            self.dilation2 = 3
        elif size == 4:
            self.dilation1 = 2
            self.dilation2 = 4
        elif size == 5:
            self.dilation1 = 2
            self.dilation2 = 5
        elif size == 6:
            self.dilation1 = 2
            self.dilation2 = 6
        elif size == 7:
            self.dilation1 = 2
            self.dilation2 = 7
        #self.dilation1 = 2#(size//8-3)//2 if (size//8-3)//2 > 0 else 1
        #self.dilation2 = 3#(size//4-3)//2 if (size//4-3)//2 >0 else 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=self.dilation1, dilation=self.dilation1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=self.dilation2, dilation=self.dilation2, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True), 
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, 1, 1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.conv4(x1+x3)
        return out

class sconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(sconv_block,self).__init__()
        self.conv = nn.Sequential(        
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class SwinNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
    # def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, **kwargs):
        super(SwinNet,self).__init__()
    
        self.Conv1 = swinconv_block(ch_in=3,ch_out=32,size=5)
        self.Conv2 = swinconv_block(ch_in=32,ch_out=64,size=5)
        self.Conv3 = swinconv_block(ch_in=64,ch_out=128,size=5)
        self.Conv4 = swinconv_block(ch_in=128,ch_out=256,size=5)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.Conv_1x1 = nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0)
        self.Conv_1x2 = nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0)
        self.Conv_1x3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        
        self.last = sconv_block(128, 64)
        self.last2 = sconv_block(64, 32)
        self.out = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        
        x41 = self.up(self.Conv_1x1(x4))
        x31 = self.up(self.last(self.Conv_1x2(x3) + x41))
        x21 = self.last2(self.Conv_1x3(x2) + x31)
 
        #out = self.out(x21)
        #out = F.interpolate(out, x.size()[2:], mode='bilinear',align_corners = False)
        out = self.up(self.out(self.up(x21)))
       

        return out