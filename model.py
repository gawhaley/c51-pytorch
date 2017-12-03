import torch
import torch.nn as nn
import torch.nn.functional as Fnn

class C51(nn.Module):   #following schematic from Mnih et al.(2015)
                        #C51 paper imitates but w/Adam & different hyperparams.
    def __init__(self,output_dim, num_atoms):
        super(C51, self).__init__()

        self.output_dim = output_dim                            #84x84x4
        self.num_atoms = num_atoms                              #   V
        self.conv1 = nn.Conv2d(4,32,kernel_size=8,stride=4)     #20x20x32
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)    #9x9x64
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)    #7x7x64
        self.lin1  = nn.Linear(3136,512)
        self.lin2  = nn.Linear(512,num_atoms)


    def forward(self, x):      #taking (batch)x4x84x84 inputs
        x = Fnn.relu(self.conv1(x))
        x = Fnn.relu(self.conv2(x))
        x = Fnn.relu(self.conv3(x))

        distribution_list = []
        for i in range(self.output_dim): #outputs output_dim # of distributions
            distribution_list.append(Fnn.relu(lin1(x)))
            distribution_list[i] = Fnn.relu(lin2(distribution_list[i]))

        return distribution_list

def prepare_env():
    return None


def train():
    return None
