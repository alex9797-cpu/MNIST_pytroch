



import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from architetctures import*



' Donwload Datsets'

mnist_trainset = datasets.MNIST(root='\data', train=True, download=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
mnist_testset = datasets.MNIST(root='\data', train=False, download=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

' Set up train and testloader'

batch_size=64

train_loader=DataLoader(mnist_trainset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(mnist_testset,batch_size=batch_size,shuffle=True)

for batch in train_loader:
    image,label=batch
    break


net= Feed_forward_net()

image=torch.reshape(image,(64,784))


out=net(image)

print(out.shape)






























