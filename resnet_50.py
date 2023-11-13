"""BASED ON https://arxiv.org/pdf/1512.03385v1.pdf"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        super(BottleNeck,self).__init__()
        self.c1=nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU())

        self.c2=nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU())

        self.c3=nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))


        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(out_channels))

    def forward(self,x):
        out=self.c1(x)
        out=self.c2(out)
        out=self.c3(out)
        out+=self.shortcut(x)
        return F.relu(out)




class ResNet_50(nn.Module):
    def __init__(self):
        super(ResNet_50,self).__init__()
        self.relu=nn.ReLU()
        self.con=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.maxpool=nn.MaxPool2d(3,2,1)

        self.Block=nn.Sequential(
            self._make_block(256,128,3),
            self._make_block(512,256,4),
            self._make_block(1024,512,6),
            self._make_block(2048,2048,3))

        self.fc=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4096,4096),
            nn.Linear(4096,4))


        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def _make_block(self,in_channels,out_channels,num_blocks):  #in_channels=256,out_channels=128
        self.blocks=[BottleNeck(in_channels//4,in_channels//4,out_channels*2)]
        for _ in range(1,num_blocks-1):
            block=BottleNeck(out_channels*2,in_channels//4,out_channels*2)
            self.blocks.append(block)

        if out_channels==in_channels:
            out=nn.Sequential(*self.blocks)
        else:
            out=nn.Sequential(*self.blocks,
                              nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2,padding=0))
        return out



    def forward(self,x):
        out=self.relu(self.con(x))
        out=self.maxpool(out)
        out=self.Block(out)
        out=self.fc(out)

        return out


if __name__=="__main__":
    model=ResNet_50()
    x=torch.randn(32,3,224,224)
    out=model(x)
    print(out)


