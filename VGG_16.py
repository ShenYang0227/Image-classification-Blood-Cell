"""BASED ON https://arxiv.org/pdf/1409.1556.pdf"""


import torch
import torch.nn as nn
import torch.nn.functional as F
device=torch.device("cuda")
class VGG_Block(nn.Module):
    def __init__(self,in_channles,out_channels):
        super(VGG_Block,self).__init__()

        self.layer1=nn.Sequential(
            nn.Conv2d(in_channles,out_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU()).to(device)

        self.layer2=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.ReLU()).to(device)

    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)

        return out





class VGG_16(nn.Module):
    def __init__(self,block_shapes):
        super(VGG_16,self).__init__()

        self.block_shapes=block_shapes
        self.blocks=[]
        for i in range(len(self.block_shapes)-1):
            block=VGG_Block(self.block_shapes[i],self.block_shapes[i+1])
            self.blocks.append(block)
            
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512,4096),
            nn.Dropout(0.3),

            nn.Linear(4096,4096),
            nn.Dropout(0.5),

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

    def forward(self,x):
        output=[x]
        for l in range(len(self.blocks)):
            out=self.blocks[l].forward(output[l])
            output.append(out)
        out=output[-1]

        out=self.fc(out)

        return out



if __name__=="__main__":

    x=torch.randn(32,3,224,224)
    block_shape=[3,64,128,256,512,512]
    model=VGG_16(block_shape)
    out=model(x)
    print(out)
