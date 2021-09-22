import torch
import argparse
from torchvision import transforms
from torchvision.datasets import MNIST,CIFAR10
from torch.utils.data import DataLoader
from torch.nn import functional as F
from VAE import VAE
import numpy as np
import matplotlib.pyplot as plt

def loss_fn(recon_x, x, mean, log_var):
    mse_loss = F.mse_loss(recon_x, x)
    # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim=1), dim=0)
    ce_loss=F.binary_cross_entropy(recon_x.view(x.size(0),-1),x.view(x.size(0),-1), reduction='mean')
    loss = mse_loss+ce_loss
    return loss, mse_loss, ce_loss

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'mnist':
        dataset = MNIST('data/mnist/', train=True, download=True, transform=transforms.ToTensor())
        dataset_test = MNIST('data/mnist/', train=False, download=True, transform=transforms.ToTensor())
        vae = VAE(in_channels=1, latent_dim=args.latent_dim, p=[2,0],l=4).to(device)
    elif args.dataset == 'cifar':
        dataset = CIFAR10('data/cifar', train=True, download=True, transform=transforms.ToTensor())
        dataset_test = CIFAR10('data/cifar', train=False, download=True, transform=transforms.ToTensor())
        vae = VAE(in_channels=3, latent_dim=args.latent_dim, p=[1,1],l=4).to(device)

    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    mse_loss_train,mse_loss_test,ce_loss_train,ce_loss_test=[],[],[],[]
    for epoch in range(args.epochs):
        vae.train()
        mse_loss_train_iter=0
        ce_loss_train_iter=0
        for iteration, (x, y) in enumerate(data_loader):
            x= x.to(device)
            recon_x, mean, log_var, z = vae(x)
            loss,mse_loss, ce_loss = loss_fn(recon_x, x, mean, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_loss_train_iter+=mse_loss
            ce_loss_train_iter+=ce_loss

            if iteration % args.freq == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:6.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader) - 1, loss.item()))

        torch.save(vae.state_dict(),'model/{}/{}_{}.pth'.format(args.dataset,epoch,args.latent_dim))
        mse_loss_train.append(mse_loss_train_iter.item())
        ce_loss_train.append(ce_loss_train_iter.item())

        vae.eval()
        mse_loss_test_iter,ce_loss_test_iter=0,0
        for iteration, (x, y) in enumerate(data_loader_test):
            x = x.to(device)
            with torch.no_grad():
                recon_x, mean, log_var, z = vae(x)
            loss, mse_loss, ce_loss = loss_fn(recon_x, x, mean, log_var)
            mse_loss_test_iter += mse_loss
            ce_loss_test_iter += ce_loss
        mse_loss_test.append(mse_loss_test_iter.item())
        ce_loss_test.append(ce_loss_test_iter.item())
        # print("Epoch {:02d}/{:02d} , Loss_train_sum {:6.4f} , Loss_test_sum {:6.4f}".format(
        #     epoch, args.epochs, loss_train_iter,loss_test_iter))

    return np.array(mse_loss_train),np.array(ce_loss_train),np.array(mse_loss_test),np.array(ce_loss_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--freq", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--dataset", type=str,default='cifar')

    args = parser.parse_args()
    mse_loss_train,ce_loss_train,mse_loss_test,ce_loss_test=main(args)

    #impact of latent size and learning rate
    # fig1 = plt.figure(1)
    # ax1 = fig1.add_subplot(111)
    # fig2 = plt.figure(2)
    # ax2 = fig2.add_subplot(111)
    # for i in [0.01,0.005, 0.001,0.0005, 0.0001]:
    #     args.learning_rate = i
    #     mse_loss_train, ce_loss_train, mse_loss_test, ce_loss_test = main(args)
    #     ax1.plot(mse_loss_train + ce_loss_train, label='learning_rate={}'.format(i))
    #     ax2.plot(mse_loss_test + ce_loss_test, label='learning_rate={}'.format(i))
    # ax1.legend()
    # ax2.legend()
    # plt.show()

    #save loss
    # np.save(r'loss\mse_loss_train1', mse_loss_train)
    # np.save(r'loss\ce_loss_train1', ce_loss_train)
    # np.save(r'loss\mse_loss_test1', mse_loss_test)
    # np.save(r'loss\ce_loss_test1', ce_loss_test)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # le1=ax1.plot(mse_loss_train,c='r',label='mse_loss_train')
    # ax2 = ax1.twinx()
    # le2=ax2.plot(ce_loss_train,c='c',label='ce_loss_train')
    # le = le1 + le2
    # labs = [l.get_label() for l in le]
    # ax1.legend(le, labs, loc=0)
    # ax1.set_xlabel('epoch')
    # ax1.set_ylabel('mse')
    # ax2.set_ylabel('ce')