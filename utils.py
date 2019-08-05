import torch
import torchvision
from torchvision import datasets, transforms


def get_data():
   
    # transform the data to tensor
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5])])


    data_train = datasets.MNIST(root = "./data/",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.MNIST(root="./data/",
                            transform = transform,
                            train = False)



    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size = 64,
                                                    shuffle = True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                batch_size = 64,
                                                shuffle = True)

    return data_loader_train, data_loader_test, (len(data_train), len(data_test))

if __name__ == '__main__':
    a,b,c = get_data()
    for idx,(x,y) in enumerate(a):
        print(x.shape,y.shape)
