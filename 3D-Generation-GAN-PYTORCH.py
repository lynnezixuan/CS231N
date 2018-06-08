import os
import sys
import shutil
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import scripts
sys.path.insert(0, '../')
import random
import scipy.ndimage as nd
import scipy.io as io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils import data
from torch.autograd import Variable
import os
import pickle
from collections import OrderedDict
from torch import optim   
from torch.autograd import Variable
from torch.utils import data

epochs = 1500
genorator_learning_rate = 0.0025
discriminator_learning_rate = 0.00005
output_size = 32
batch_size = 256

def save_voxels(voxels, path,epoch):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig(path + '/{}.png'.format(str(epoch)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(epoch)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)
        

def save_model(path, epoch, G, G_optimizer, D, D_optimizer):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/Generator" + str(epoch) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optimizer" + str(epoch) + ".pkl", "wb") as f:
        torch.save(G_optimizer.state_dict(), f)
    with open(path + "/Discriminator" + str(epoch) + ".pkl", "wb") as f:
        torch.save(D.state_dict(), f)
    with open(path + "D_optimizer" + str(epoch) + ".pkl", "wb") as f:
        torch.save(D_optimizer.state_dict(), f)
        
def z_generator():

    Z = (torch.Tensor(batch_size, 200).normal_(0, 1))
    Z = Z.cuda()

    return Variable(Z)

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.cube_len = 32
        padd = (0, 0, 0)
        bias = False
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(200, self.cube_len*8, kernel_size=4, stride=2, bias=bias, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len,1, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        out = x.view(-1, 200, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cube_len = 32

        padd = (0,0,0)
        bias = False
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=bias, padding=padd),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 1, 32, 32, 32)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
    

class ShapeNetDataset(data.Dataset):

    def __init__(self, root):
        self.root = root
        self.listdir = os.listdir(self.root)

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.load(f)
            return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)
  

dsets = ShapeNetDataset('data/train/chair/')
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=batch_size, shuffle=True, num_workers=1)

D = Discriminator()
D.cuda()
G = Generator()
G.cuda()

D_optimizer = optim.Adam(D.parameters(), lr=0.00005, betas=(0.5, 0.5))
G_optimizer = optim.Adam(G.parameters(), lr=0.0025, betas=(0.5, 0.5))
criterion = nn.BCELoss()

pickle_path = './output/pickle/'
loss_result = open("results.txt","w")


for epoch in range(epochs):
        for i, X in enumerate(dset_loaders):

            X = Variable(X.cuda())
            if X.size()[0] != int(batch_size):
                continue
            D.zero_grad()
            G.zero_grad()
            Z = z_generator()
            
            ground_truth = Variable(torch.ones(batch_size).cuda())
            fake_labels = Variable(torch.zeros(batch_size).cuda())

            d_real = D(X)
            d_real_loss = criterion(d_real, ground_truth)

            fake = G(Z)
            d_fake = D(fake)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            
            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))
            
            if d_total_acu.data[0]<0.95:
                D.zero_grad()
                d_loss.backward()
                D_optimizer.step()
            
            
            Z = z_generatorZ()
            fake = G(Z)
            d_fake = D(fake)
            g_loss = criterion(d_fake, real_labels)

           
            g_loss.backward()
            G_optimizer.step()
            
            
       
        print('Epoch{} , D_loss : {:.4}, G_loss : {:.4}'.format(epoch, d_loss.data[0], g_loss.data[0]))
        
        loss_result.write('Epoch{} , D_loss : {:.4}, G_loss : {:.4}'.format(epoch,d_loss.data[0], g_loss.data[0]))                                                 
        loss_result.write('\n')
                   
        if (epoch + 1) % 10 == 0:

            samples = fake.cpu().data[:8].squeeze().numpy()

            image_path = './output/imag/' 
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            save_voxels(samples, image_path,epoch)

        if (epoch + 1) % 100 == 0:
            pickle_save_path = './output/pickle/'
            if not os.path.exists(pickle_save_path):
                os.makedirs(pickle_save_path)
            save_model(pickle_save_path,epoch, G, G_optimizer, D, D_optimizer)