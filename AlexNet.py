"""THIS NEURAL NETWORK Based ON
ImageNet https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Alex_Net(nn.Module):
    def __init__(self):
        super(Alex_Net,self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=1),
            nn.MaxPool2d(3, 2, padding=0),
            nn.ReLU())

        self.conv2=nn.Sequential(
            nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(3, 2),
            nn.ReLU())

        self.conv3=nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1))

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*6*256,4096),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.Dropout(0.3),
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
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.fc(out)

        return out


if __name__=="__main__":

    x=torch.randn(32,3,224,224)

    model=Alex_Net()
    out=model(x)
    print(out)





