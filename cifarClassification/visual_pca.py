import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from resnet import * #resnet
#from vgg import * #vgg
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', default = 'True',
                    help='resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')


net = ResNet18() #using resnet
#net = VGG('VGG16') #using vgg
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/resnet_ckpt_lr=0.05.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

net.eval()
height = len(testloader)
width = 10
X = np.zeros((height, width))
label = np.zeros((height, 1))
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
    X[batch_idx, :] = outputs.detach().cpu().numpy()
    label[batch_idx, :] = targets.cpu().numpy()
print('Finish loading')

pca = decomposition.PCA(n_components=2)
Y = pca.fit_transform(X)

plt.scatter(Y[:, 0], Y[:, 1], c=label, cmap=plt.cm.Spectral)
plt.colorbar()
plt.title("PCA of resnet18, lr=0.05")
plt.show()