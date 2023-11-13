import copy
import time

import torch
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from AlexNet import Alex_Net
from VGG_16 import VGG_16
from GoogLeNet import GoogLeNet
from resnet_18 import ResNet_18
from resnet_50 import ResNet_50
def train_valid_data_process():   #split randomly the training sets and valid sets
    train_transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomRotation(68),
            transforms.RandomGrayscale(0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ]
    )
    train_data = datasets.ImageFolder(root="./dataset2-master/images/TRAIN", transform=train_transformer)
    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_loader=Data.DataLoader(dataset=train_data,batch_size=32,shuffle=True,num_workers=2)
    valid_loader=Data.DataLoader(dataset=val_data,batch_size=32,shuffle=True,num_workers=2)
    return train_loader,valid_loader



def train(model,train_loader,valid_loader,num_epochs):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.CrossEntropyLoss()


    model=model.to(device)
    best_model_wts=copy.deepcopy(model.state_dict())

    best_acc=0.0
    train_loss_all=[]
    valid_loss_all=[]

    train_acc_all=[]
    valid_acc_all=[]

    since=time.time()
    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch,num_epochs-1))
        print("-"*10)
        train_loss=0.0
        train_corrects=0.0
        train_num=0

        valid_loss = 0.0
        valid_corrects = 0.0
        valid_num=0

        for step,(b_x,b_y) in enumerate(train_loader):
            batch_x=b_x.to(device)
            batch_y=b_y.to(device)


            model.train()
            output=model(batch_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*batch_x.size(0)

            train_corrects += torch.sum(pre_lab==batch_y.data)

            train_num += batch_x.size(0)

        for step, (b_x,b_y) in enumerate(valid_loader):
            batch_x = b_x.to(device)
            batch_y = b_y.to(device)

            model.eval()
            output=model(batch_x).to(device)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, batch_y)

            valid_loss += loss.item()*batch_x.size(0)
            valid_corrects+=torch.sum(pre_lab==batch_y.data)

            valid_num+=batch_x.size(0)

        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)

        valid_loss_all.append(valid_loss/valid_num)
        valid_acc_all.append(valid_corrects.double().item()/valid_num)

        print("{} Train Loss:{:.4f} Train Acc:{:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} Valid Loss:{:.4f} Valid Acc:{:.4f}".format(epoch,valid_loss_all[-1],valid_acc_all[-1]))

        if valid_acc_all[-1]>best_acc:
            best_acc=valid_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
        time_use=time.time()-since
        print("Training Time is: {:.0f}m{:.0f}s".format(time_use//60,time_use%60))


    torch.save(best_model_wts,"./ResNet_50_best_model.pt")

    train_process=pd.DataFrame(data={"epoch":range(num_epochs),
                                         "train_loss_all":train_loss_all,
                                         "valid_loss_all":valid_loss_all,
                                         "train_acc_all":train_acc_all,
                                         "valid_acc_all":valid_acc_all
                                         })

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,"ro-",label="train loss")
    plt.plot(train_process["epoch"],train_process.valid_loss_all,"bs-",label="valid loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train accuracy")
    plt.plot(train_process["epoch"], train_process.valid_acc_all, "bs-", label="valid accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.savefig("./ResNet_50_result")
    plt.show()

def main():
    model=ResNet_50()
    train_data,valid_data=train_valid_data_process()

    train_process=train(model,train_data,valid_data,num_epochs=50)
    matplot_acc_loss(train_process)

if __name__=="__main__":
    main()








