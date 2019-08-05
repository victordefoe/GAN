# this is refer from the 


import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

from utils import get_data
from models import generator, discriminator


def show_data():
    data_loader_train, data_loader_test, total_num = get_data()
    batch_images, batch_labels = next(iter(data_loader_train)) # tensors

    #show some examples of data : can be comment
    img = torchvision.utils.make_grid(batch_images)

    img = img.numpy().transpose(1,2,0)
    std = [0.5,0.5,0.5]
    mean = [0.5,0.5,0.5]
    img = img*std+mean
    print([labels[i] for i in range(64)])
    plt.imshow(img)


def plot_images(G_NET, inputs_noise, n_images, save_id):
    """
        G_NET: a trained network
        inouts_noise: a standard noise form only for getting the shape, (Tensor)
        n_images: how many images need to generate for showing
        save_id: imply which directory you put it in ; (int)
    """
    noise_shape = inputs_noise.shape[-1]  #take the last dimension
    # create noise images
    examples_noise = np.random.uniform(-1, 1, size=[n_images, noise_shape])
    examples_noise = torch.Tensor(examples_noise).cuda()
    samples = G_NET(examples_noise)
    print(samples.shape)
    samples = np.squeeze(samples).detach().cpu().numpy()


    fig_id = 0
    save_path = './generated_data/%d/' % save_id
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for img in samples:
        # img = Image.fromarray(sample)
        plt.imshow(img.reshape(28,28), cmap='Greys_r')
        plt.savefig(save_path + "%d.png" % (fig_id))
        # im.save(save_path + "%d.png" % (fig_id))
        fig_id += 1


# cost = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())


def train():
    # get the data from pytorch tool itself
    data_loader_train, data_loader_test, total_num = get_data()

    # set parameters
    print_interval = 1
    save_interval  = 1
    n_epochs = 20
    batch_size = 64
    noise_size = 100
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    # device = 'cuda0'

    # dcgan need two optimizers and losses


    # creating generator, fake the various data base on different noise initial condition
    g_net = generator(noise_size)
    # g_net.to(device)
    d_net = discriminator()
    # d_net.to(device)
    if torch.cuda.is_available() == True:
        g_net = g_net.cuda()
        d_net = d_net.cuda()
    else:
        assert False, 'You should run it on GPU, else use CPU Version!'

    optim_betas = (0.9, 0.999)
    d_learning_rate = 2e-4 
    g_learning_rate = 2e-4

    d_optimizer = optim.Adam(g_net.parameters(), lr=d_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(d_net.parameters(), lr=g_learning_rate, betas=optim_betas)
    steps = 0
    folderid = 0
    for epoch in range(n_epochs):
        for batch_i in range(total_num[0]//batch_size):
            steps += 1
            g_net.zero_grad()
            d_net.zero_grad()
            ### training process
            # read the data batch
            batch_images, batch_labels = next(iter(data_loader_train))
            # batch_images = batch_images.to(device)
            batch_images = batch_images.cuda()
            batch_noise = torch.Tensor(np.random.uniform(-1, 1, size=(batch_size, noise_size)))
            # batch_noise = batch_noise.to(device)
            batch_noise = batch_noise.cuda()
            # generate the fake data
            g_data = g_net(batch_noise)
            # sent the generation to the discriminator network
            d_fake_data = d_net(g_data)
            d_real_data = d_net(batch_images)

            # Calculated sevral kinds of loss
            loss_d_fake = criterion(d_fake_data, torch.zeros((16,1)).cuda())
            loss_d_real = criterion(d_real_data, torch.ones((16,1)).cuda())
            loss_d = loss_d_fake + loss_d_real
            loss_g = criterion(d_fake_data, torch.ones((16,1)).cuda())
            loss_d.backward(retain_graph=True)
            loss_g.backward()
            d_optimizer.step()
            g_optimizer.step()


        if epoch % save_interval == 0:
            torch.save(g_net.state_dict(), './weights/GAN_G_Mnist_E%d_B%d.pth' % (epoch, batch_i))
            torch.save(d_net.state_dict(), './weights/GAN_D_Mnist_E%d_B%d.pth' % (epoch, batch_i))
        if epoch % print_interval == 0:
            print("Epoch [%d]/[%d]:Generator Loss: %.4f Discriminator Loss:%.4f(Real: %s, Fake: %s)"
                 % (epoch, n_epochs, loss_g, loss_d, loss_d_real, loss_d_fake))
            plot_images(g_net,batch_noise, 25, folderid)
            folderid += 1
            print('Saved images')



            

if __name__ == '__main__':
    train()
