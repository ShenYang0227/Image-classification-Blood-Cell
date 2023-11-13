
import torch
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as Data
from sklearn import metrics

from GoogLeNet import GoogLeNet
from resnet_18 import ResNet_18
from resnet_50 import ResNet_50
def test(model):

    test_transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomRotation(68),
            transforms.RandomGrayscale(0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    test_data = datasets.ImageFolder("./dataset2-master/images/TEST", transform=test_transformer)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=2)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)

    test_corrects=0.0
    test_num=0
    with torch.no_grad():
        for data_x,data_y in test_loader:
            data_x=data_x.to(device)
            data_y=data_y.to(device)

            model.eval()
            output=model(data_x)

            pre_lab=torch.argmax(output,dim=1)
            test_corrects+=torch.sum(pre_lab==data_y)
            test_num+=data_x.size(0)

    test_acc=test_corrects.double().item()/test_num
    print("the accurate rate is {:.3f}".format(test_acc))
    f1=metrics.f1_score(data_y,pre_lab,average="samples")
    print("the f1 score is {:.3f}".format(f1))


if __name__=="__main__":
    model=ResNet_50()
    model.load_state_dict(torch.load("ResNet_50_best_model.pt"))

    test(model)






