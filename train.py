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
from visual_loss import Visualizer
from torchnet import meter


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
        plt.imsave(save_path + "%d.png" % (fig_id), img.reshape(28,28), cmap='Greys_r')
        # plt.imshow(img.reshape(28,28), cmap='Greys_r')
        # plt.plot(img.reshape(28,28), cmap='Grey_r')
        # plt.savefig(save_path + "%d.png" % (fig_id))
        # im.save(save_path + "%d.png" % (fig_id))
        fig_id += 1


# cost = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())


vis = Visualizer(env='my_wind')#为了可视化增加的内容
loss_meter = meter.AverageValueMeter()#为了可视化增加的内容


def train():
    # get the data from pytorch tool itself
    data_loader_train, data_loader_test, total_num = get_data()

    # set parameters
    reuse = True
    print_interval = 100 # how many batches for one show(loss)
    save_interval  = 1 # how many epoches for one save(model and images)
    n_epochs = 10
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
    d_learning_rate = 2e-5
    g_learning_rate = 2e-5
    G_Step = 1
    D_Step = 1

    d_optimizer = optim.Adam(g_net.parameters(), lr=d_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(d_net.parameters(), lr=g_learning_rate, betas=optim_betas)
    if reuse == True:
        try:
            read_epoch = 15
            read_batch = 936
            checkpoint_G = torch.load('./weights/GAN_G_Mnist_E%d_B%d.pth' % (read_epoch, read_batch))
            checkpoint_D = torch.load('./weights/GAN_D_Mnist_E%d_B%d.pth' % (read_epoch, read_batch))
            g_net.load_state_dict(checkpoint_G)
            d_net.load_state_dict(checkpoint_D)
            print('Loading the pretrained model- Epoch%d, Batch%d' % (read_epoch, read_batch))
        except:
            print("Read PTH files failed!")


    folderid = 0
    for epoch in range(n_epochs):
        loss_meter.reset()  # 为了可视化增加的内容
        for batch_i in range(total_num[0]//(batch_size * (D_Step))):
            for batch_d in range(D_Step):
                d_net.zero_grad()
            ### training process
            # read the data batch
                d_batch_images, d_batch_labels = next(iter(data_loader_train))
                d_batch_noise = torch.Tensor(np.random.uniform(-1, 1, size=(batch_size, noise_size))).cuda()
                d_batch_images = d_batch_images.cuda()

                d_g_data = g_net(d_batch_noise).detach()
                d_fake_decision = d_net(d_g_data)
                d_real_decision = d_net(d_batch_images)
                # Calculated sevral kinds of loss
                loss_d_fake = criterion(d_fake_decision, torch.zeros((batch_size, 1)).cuda())
                loss_d_real = criterion(d_real_decision, torch.ones((batch_size, 1)).cuda())
                loss_d = loss_d_fake + 10*loss_d_real
                # loss_d_fake.backward()
                # loss_d_real.backward()
                loss_d.backward()
                d_optimizer.step()

            for batch_d in range(G_Step):
                g_net.zero_grad()

                g_batch_noise = torch.Tensor(np.random.uniform(-1, 1, size=(batch_size, noise_size))).cuda()

                g_g_data = g_net(g_batch_noise)
                g_fake_decision = d_net(g_g_data)
                loss_g = criterion(g_fake_decision, torch.ones((batch_size, 1)).cuda())
                loss_g.backward()

                g_optimizer.step()

            # generate the fake data

            # sent the generation to the discriminator network



            loss_meter.add(loss_d.cpu().data.numpy()) # visualized
            vis.plot_many_stack({'train_loss': loss_meter.value()[0]})

            if batch_i % print_interval == 0:
                print("Epoch %d/%d - batch [%d]/[%d]:Generator Loss: %.4f Discriminator Loss:%.4f(Real: %.4f, Fake: %.4f)"
                      % (epoch, n_epochs, batch_i, total_num[0]//(batch_size * (D_Step)), loss_g, loss_d, loss_d_real, loss_d_fake))


        if epoch % save_interval == 0:
            torch.save(g_net.state_dict(), './weights/GAN_G_Mnist_E%d_B%d.pth' % (epoch, batch_i))
            torch.save(d_net.state_dict(), './weights/GAN_D_Mnist_E%d_B%d.pth' % (epoch, batch_i))
            print('Saved model ----- ')
            plot_images(g_net,g_batch_noise, 25, folderid)
            folderid += 1
            print('Saved images ----- ')





if __name__ == '__main__':
    train()
