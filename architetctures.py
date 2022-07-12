
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple feed forwrad neural network

class Feed_forward_net(nn.Module):

    def __init__(self):
        super(Feed_forward_net, self).__init__()

        self.layer1=nn.Linear(28*28,200)
        self.layer2=nn.Linear(200,200)
        self.layer3=nn.Linear(200,10)




    def forward(self,x):

        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))

        return(x)








