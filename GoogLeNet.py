
"""BASED ON
https://arxiv.org/pdf/1409.4842v1.pdf"""


import torch
import torch.nn as nn
import torch.nn.functional as F
class Inception(nn.Module):
    def __init__(self,in_channels,p1_channels,p2_channels,p3_channels,p4_channels):
        super(Inception,self).__init__()
        self.relu=nn.ReLU()
        self.p1=nn.Conv2d(in_channels,p1_channels,kernel_size=1,stride=1,padding=0)

        self.p2_1=nn.Conv2d(in_channels,p2_channels[0],kernel_size=1,stride=1,padding=0)
        self.p2_2=nn.Conv2d(p2_channels[0],p2_channels[1],kernel_size=3,stride=1,padding=1)

        self.p3_1=nn.Conv2d(in_channels,p3_channels[0],kernel_size=1,stride=1,padding=0)
        self.p3_2=nn.Conv2d(p3_channels[0],p3_channels[1],kernel_size=5,stride=1,padding=2)

        self.p4_1=nn.MaxPool2d(3,1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,p4_channels,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        p1=self.relu(self.p1(x))
        p2=self.relu(self.p2_2(self.relu(self.p2_1(x))))
        p3=self.relu(self.p3_2(self.relu(self.p3_1(x))))
        p4=self.relu(self.p4_2(self.relu(self.p4_1(x))))


        return torch.cat((p1,p2,p3,p4),dim=1)
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.b1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,padding=1))

        self.b2=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1))

        self.b3=nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(3,2,1))

        self.b4=nn.Sequential(
            Inception(480,192,(96,208),(16,48),64),
            Inception(512,160,(112,224),(24,64),64),
            Inception(512,128,(128,256),(24,64),64),
            Inception(512,112,(128,288),(32,64),64),
            Inception(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(3,2,1))


        self.b5=nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(192,384),(48,128),128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024,4)
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

                elif isinstance(m,nn.Linear):
                    nn.init.normal_(m.weight,0,0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias,0)


    def forward(self,x):
        out=self.b1(x)
        out=self.b2(out)
        out=self.b3(out)
        out=self.b4(out)
        out=self.b5(out)
        return out

if __name__=="__main__":

    x=torch.randn(32,3,224,224)

    model=GoogLeNet()
    out=model(x)
    print(out)


