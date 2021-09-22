import torch
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torchvision import transforms
from torchvision.datasets import MNIST,CIFAR10
from torch.utils.data import DataLoader
from VAE import VAE
from train import loss_fn
from visualization import pca,tsne

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,default='cifar')
parser.add_argument("--latent_dim", type=int, default=128)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'mnist':
    dataset_test = MNIST('data/mnist/', train=False, download=True, transform=transforms.ToTensor())
    vae = VAE(in_channels=1, latent_dim=args.latent_dim, p=[2,0], l=4).to(device)
elif args.dataset == 'cifar':
    dataset_test = CIFAR10('data/cifar', train=False, download=True, transform=transforms.ToTensor())
    vae = VAE(in_channels=3, latent_dim=args.latent_dim, p=[1,1],l=4).to(device)

data_loader_test = DataLoader(dataset=dataset_test, batch_size=1024, shuffle=False)

vae.load_state_dict(torch.load('model/{}/29_128.pth'.format(args.dataset)))
vae.eval()
for iteration, (x, y) in enumerate(data_loader_test):
    x = x.to(device)
    x_encoded=vae.visualize(x)
    if args.dataset=='mnist':
        pca(x.view(-1, 28 * 28), y)
        tsne(x.view(-1, 28 * 28), y)
        pca(x_encoded, y)
        tsne(x_encoded, y)
    else:
        pca(x.view(-1, 32 * 32 * 3), y)
        tsne(x.view(-1, 32 * 32 * 3), y)
        pca(x_encoded, y)
        tsne(x_encoded, y)

    recon_x, mean, log_var, z = vae(x)
    loss,recons_loss, kld_loss = loss_fn(recon_x, x, mean, log_var)

    plt.figure(figsize=(8, 8))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        a=x[i,:]
        if args.dataset=='mnist':
            plt.imshow(a.view(28, 28).mul(255).byte().cpu().numpy())
        else:
            plt.imshow(a.mul(255).byte().cpu().numpy().transpose((1, 2, 0)))
        plt.axis('off')
    plt.savefig('figs/{}/{}.png'.format(args.dataset,iteration), dpi=300)
    plt.clf()
    plt.close('all')

    plt.figure(figsize=(8, 8))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        a = recon_x[i,:]
        if args.dataset == 'mnist':
            plt.imshow(a.view(28, 28).mul(255).byte().cpu().numpy())
        else:
            plt.imshow(a.mul(255).byte().cpu().numpy().transpose((1, 2, 0)))
        plt.axis('off')
    plt.savefig('figs/{}/{}_recon.png'.format(args.dataset,iteration), dpi=300)
    plt.clf()
    plt.close('all')




