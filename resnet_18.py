"FROM https://arxiv.org/pdf/1512.03385v1.pdf"



import torch
import torch.nn as nn
import torch.nn.functional as F
class Residual_Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Residual_Block,self).__init__()
        self.relu=nn.ReLU()
        self.c1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.c2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        self.shortcut=nn.Sequential()
        if in_channels!=out_channels:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
                                        nn.BatchNorm2d(out_channels))

    def forward(self,x):
        identity=x
        out=self.relu(self.bn1(self.c1(x)))
        out=self.bn2(self.c2(out))

        out+=self.shortcut(identity)
        return self.relu(out)



class ResNet_18(nn.Module):
    def __init__(self):
        super(ResNet_18,self).__init__()
        self.c1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.maxpool=nn.MaxPool2d(3,2,padding=1)

        self.layers=nn.Sequential(
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            self._make_layer(512,512,2)
        )



        self.fc=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1*1*512,4096),
            nn.Linear(4096,4)
        )

        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
        #
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias,0)
        #
        #     elif isinstance(m,nn.Linear):
        #         nn.init.normal_(m.weight,0,0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias,0)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        self.layer = []
        for _ in range(num_blocks-1):
            layer = Residual_Block(in_channels, in_channels)
            self.layer.append(layer)

        if in_channels==out_channels:
            out=nn.Sequential(*self.layer)

        else:

            out = nn.Sequential(
                *self.layer,
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        return out


    def forward(self,x):
        out=self.c1(x)
        out=self.maxpool(out)
        out=self.layers(out)
        out=self.fc(out)
        return out

if __name__=="__main__":
    model=ResNet_18()
    x=torch.randn(32,3,224,224)
    out=model(x)
    print(out.shape)
    print(out)