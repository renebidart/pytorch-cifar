'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
parser.add_argument("--bs", type=int, default=256)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument('--save_path', metavar='DIR', default=".", type=str, help='path to save output')

parser.add_argument('--model_type', type=str)
parser.add_argument("--num_classes", type=int, default=10)

parser.add_argument("--channel_factor", type=int)
parser.add_argument("--ws_factor", type=int, default=1)
parser.add_argument("--ws_sp_factor", type=int, default=1)
parser.add_argument("--ws_ch_factor", type=int, default=1)
parser.add_argument("--downsample_repeat", action='store_true')

parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_hidden", type=int, default=256)
parser.add_argument('--norm', type=str)
parser.add_argument('--spatial', type=str)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'CIFAR10': 
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    assert args.num_classes == 10
elif args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    assert args.num_classes == 100
    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=2*args.bs, shuffle=False, num_workers=2)

if args.model_type=='resnet18':
    net = ResNetLight(BasicBlockLight, [2, 2, 2, 2], ws_factor=args.ws_factor, downsample_repeat=args.downsample_repeat,
                      channel_factor=args.channel_factor, num_classes=args.num_classes, cifar=True)
elif args.model_type=='resnet34':
    net = ResNetLight(BasicBlockLight, [3, 4, 6, 3], ws_factor=args.ws_factor, downsample_repeat=args.downsample_repeat,
                      channel_factor=args.channel_factor, num_classes=args.num_classes, cifar=True)
elif args.model_type=='resnet50':
    net = ResNetLight(BottleneckLight, [3, 4, 6, 3], ws_factor=args.ws_factor, downsample_repeat=args.downsample_repeat,
                      channel_factor=args.channel_factor, num_classes=args.num_classes, cifar=True)
elif args.model_type=='resnet101':
    net = ResNetLight(BottleneckLight, [3, 4, 23, 3], ws_factor=args.ws_factor, downsample_repeat=args.downsample_repeat,
                      channel_factor=args.channel_factor, num_classes=args.num_classes, cifar=True)
elif args.model_type=='resnet152':
    net = ResNetLight(BottleneckLight, [3, 8, 36, 3], ws_factor=args.ws_factor, downsample_repeat=args.downsample_repeat,
                      channel_factor=args.channel_factor, num_classes=args.num_classes, cifar=True)
elif args.model_type=='resnet18bn':
    net = ResNetLight(BasicBlockLightBN, [2, 2, 2, 2], ws_factor=args.ws_factor, downsample_repeat=args.downsample_repeat,
                      channel_factor=args.channel_factor, num_classes=args.num_classes, cifar=True)
elif args.model_type=='resnet18std':
    net = ResNet18()
    
elif args.model_type=='xception':
    net = XceptionLight(spatial_bf=args.ws_sp_factor, channel_bf=args.ws_ch_factor, 
                        total_f=args.channel_factor, cifar=True, num_classes=args.num_classes, in_chans=3)
elif args.model_type=='vsxception':
    net = VSXception(spatial_bf=args.ws_sp_factor, channel_bf=args.ws_ch_factor, 
                        total_f=args.channel_factor, num_classes=args.num_classes, in_chans=3)
elif args.model_type=='wsmobilenetv2':
    net = LightMobileNetV2(ws_sp_factor=args.ws_sp_factor, num_classes=args.num_classes)
elif args.model_type=='mobilenetv2':
    net = MobileNetV2(num_classes=args.num_classes)
    
elif args.model_type=='albert':
    net = ALBERT(n_layers=args.n_layers, n_hidden=args.n_hidden, norm=args.norm, spatial=args.spatial, 
                 cifar=True, n_classes=args.num_classes)
    
net.to(device)
net.train()
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.save_path + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

# if 'xception' in args.model_type:
#     args.lr = .015
#     epochs = 500
#     optimizer = optim.Adam(net.parameters(), lr=args.lr)
#     scheduler = MultiStepLR(optimizer, milestones=[150, 250, 350, 450], gamma=0.2)
# else:
epochs = 350
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'num_params': num_params
        }
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        torch.save(state, args.save_path + '/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
