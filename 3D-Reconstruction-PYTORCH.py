import os
import sys
import shutil
import time
import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import scripts
sys.path.insert(0, '../')
import random as random
%load_ext autoreload
%autoreload 2

import scipy.ndimage as nd
import scipy.io as io
import matplotlib
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
from collections import OrderedDict
from glob import glob
from torch.autograd import Variable
from torch.utils import data as dat
from torch import optim

epochs = 1500
genorator_learning_rate = 0.0025
discriminator_learning_rate = 0.00005
output_size = 32
batch_size = 32
surfaces = 'data/surfaces/train/chair'
valid_surfaces = 'data/surfaces/valid/chair'
data = 'data/train/chair'


def save_voxels(voxels, path, epoch):
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


def grab_files_surfaces(surface_dir, voxel_dir): 
    surface_dir+='/'
    voxel_dir+='/'
    files = glob(surface_dir+'*')
    voxels = [ v.split('/')[-1].split('_')[-1] for v in glob(voxel_dir + '*')]
    temp = []
    for f in files: 
        if f.split('/')[-1].split('_')[-2] + '.npy' not in voxels: continue
        temp.append(f)
    return temp



class DoubleShapeNetDataset(dat.Dataset):

    def __init__(self, files):
        self.listdir = files

    def __getitem__(self, index):
        with open(self.listdir[index], "rb") as f:
            name = f.name.split('/')[-1].split('_')[-2] + '.npy'
            name = os.path.join('data/train/chair', name)
            volume = torch.from_numpy(np.load(f)).float()
            ground_truth = torch.from_numpy(np.load(name)).float()
            return torch.FloatTensor(volume), torch.FloatTensor(ground_truth)

    def __len__(self):
        return len(self.listdir)

def z_generator():

    Z = (torch.Tensor(batch_size, 200).normal_(0, 1))
    Z = Z.cuda()

    return Variable(Z)


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


class surface_VAE(nn.Module):
    def __init__(self):
        super(surface_VAE, self).__init__()

        # encoder
        self.e1 = nn.Conv3d(1, 32, 4, 2, 1)

        self.e2 = nn.Conv3d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm3d(64)

        self.e3 = nn.Conv3d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm3d(128)

        self.e4 = nn.Conv3d(128, 256, 4, 2, 1)
        self.bn4 = nn.BatchNorm3d(256)

        self.fc1 = nn.Linear(2048, 200)
        self.fc2 = nn.Linear(2048, 200)
        
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.2)

    def encode(self, x):
        h1 = self.leakyrelu(self.e1(x))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h4 = h4.view(-1, 2048)

        return self.fc1(h4), self.tanh(self.fc2(h4))

    def forward(self, image):
        image = image.view(32,1,32,32,32)
        mean,std = self.encode(image)
        
        return mean, std
    
LAMBDA = 10.0
def calc_gradient_penalty(D, real_data, fake_data):
    alpha = np.random.rand(batch_size, 1, 1, 1)
    alpha = torch.from_numpy(alpha*np.ones(real_data.size())).float()
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



D = Discriminator()
G = Generator()
surface_VAE_model = surface_VAE()
D_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
G_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
VAE_solver = optim.Adam(surface_VAE_model.parameters(), lr=1e-4, betas=(0.5, 0.9))
one = torch.FloatTensor([1])
mone = one * -1
mone = mone.cuda()
one = one.cuda()
D.cuda()
G.cuda()
criterion = nn.BCELoss()


files = grab_files_surfaces(surfaces, data)
surfaces_dsets = DoubleShapeNetDataset(files)
surfaces_dset_loaders = torch.utils.data.DataLoader(surfaces_dsets, batch_size=batch_size, shuffle=True, num_workers=1)


torch.set_default_tensor_type('torch.cuda.FloatTensor')
results_loss_rec = open("results_loss_rec.txt","w")
for epoch in range(epochs):
        for i, X_tuple in enumerate(surfaces_dset_loaders):
            X = Variable(X_tuple[0].cuda())
            y = Variable(X_tuple[1].cuda())
            if X.size()[0] != int(batch_size):
                continue

            surface_VAE_model = surface_VAE().cuda()
            
            surface_VAE_model.zero_grad()
            G.zero_grad()
            D.zero_grad()
            
            mean, std = surface_VAE_model(X)
            eps = torch.Tensor(batch_size, 200).normal_(0,1)
            sample_Z = Variable((eps*std.data + mean.data)).cuda()
            fake_sample = G(sample_Z)
            
            loss_fn = torch.nn.MSELoss()
            VAE_loss = loss_fn(fake_sample, y)
            kl_loss = torch.mean(-std +.5*(-1.+torch.exp(2.*std)+mean*mean))
            v_loss = VAE_loss + kl_loss
            v_loss.backward()
            
            
            norm_z = z_generator()
            fake_norm = G(norm_z)
            
            d_real = D(X)
            d_real = d_real.mean()
            d_real.backward(mone)

            fake = G(norm_z)
            d_fake = D(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one)
            
            gradient_penalty = calc_gradient_penalty(D, X.data, fake.data)
            gradient_penalty.backward()
            d_loss = d_fake - d_real + gradient_penalty
            Wasserstein_D = d_real - d_fake
            D_optimizer.step()
            VAE_solver.step()
            G_optimizer.step()
            

        print('Epoch{} , D_loss : {:.4}, G_loss : {:.4}, VAE_loss : {:.4}'.format(epoch, d_loss.data[0], g_loss.data[0], VAE_loss.data[0]))
        results_loss_rec.write('Epoch{} , D_loss : {:.4}, G_loss : {:.4}, VAE_loss : {:.4}'.format(epoch, d_loss.data[0], g_loss.data[0], VAE_loss.data[0]))                                                 
        results_loss_rec.write('\n')
        
        
        
            
        if (epoch + 1) % 10 == 0:

            samples = fake.cpu().data[:8].squeeze().numpy()

            image_path = './output1/imag/'
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            save_voxels(samples, image_path, epoch)

        if (epoch + 1) % 100 == 0:
            pickle_save_path = './output1/pickle/'
            if not os.path.exists(pickle_save_path):
                os.makedirs(pickle_save_path)
            save_model(pickle_save_path, epoch, G, G_optimizer, D, D_optimizer)
