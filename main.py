import os
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import wandb

from nshb import NSHB
from models.resnet import resnet18
from models.wideresnet import WideResNet28_10
from utils import progress_bar

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if args.method in ["lr", "hybrid", "cosine", "poly", "exp"]:
            last_lr = scheduler.get_last_lr()[0]
            wandb.log({'last_lr': last_lr})

    training_acc = 100.*correct/total
    wandb.log({'training_acc': training_acc,
               'training_loss': train_loss/(batch_idx+1)})

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    wandb.log({'accuracy': acc})

def polynomial_beta(initial_beta, epoch, T, p):
    decayed_beta = initial_beta * (1 - epoch / T) ** p
    return decayed_beta

def norm_work(norm_list, norm):
    norm_list.append(norm)
    average = sum(norm_list) / len(norm_list)
    return average

def next_gamma(power, M, m):
    #power means the power of polynomial decay
    #M means the number of epochs
    #m means the current epoch
    top = (M - m)
    bottom = (M - (m-1))
    gamma = (top/bottom) ** power
    return gamma

def lr_poly(lr, M, m, power):
    gamma = (M - m)/(M - (m - 1))
    gamma = gamma ** (power / 2)
    next_lr = lr * gamma
    return next_lr

def batch_poly(batch, M, m, power, balance):
    gamma = (M - (m - 1))/(M - m)
    gamma = gamma ** (power * balance)
    next_batch = batch * gamma
    return next_batch

def next_delta_SHB(lr, batch, beta, C, K):
    #lr is current learning rate
    #batch is current batch size
    #beta is current momentum factor
    #C means the variance of stochastic gradient (second order)
    #K means the upper bound of gradient norm (first order)
    beta_hat = (beta * (beta ** 2 - beta + 1)) / ((1 - beta) ** 2)
    naka = C / batch + beta_hat * ((C / batch) + K ** 2)
    delta = lr * np.sqrt(naka)
    return delta

def next_delta_NSHB(lr, batch, beta, C, K):
    naka = C / batch + (4 * (beta ** 2) * ((C / batch) + K ** 2))
    delta = lr * np.sqrt(naka)
    return delta

def solve_3eq(maru, beta):
    #maru is in coefficient for 3eq
    #beta is current momentum factor
    roots = np.roots([1, (-1 * (1 + maru)), (1 + (2 * maru)), (-1 * maru)])
    real_roots = roots[np.isreal(roots)].real
    filtered_roots = [root for root in real_roots if root <= beta]
    next_beta = max(max(filtered_roots), 0)
    return next_beta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default="CIFAR100", type=str, help="dataset name")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=32, type=int, help='training batch size')
    parser.add_argument('--epochs', default=200, type=int, help="the number of epochs")
    parser.add_argument('--decay_epoch', default=40, type=int, help="the number of epochs to decay leraning rate")
    parser.add_argument('--power', default=0.9, type=float, help="polinomial or exponential power")
    parser.add_argument('--method', default="constant", type=str, help="constant, lr, batch, beta-step, beta-poly, batch-poly, beta-batch-poly, beta-lr-poly, lr-batch-poly, hybrid, poly, cosine, exp")
    parser.add_argument('--optimizer', default="shb", type=str, help="sgd, shb, nshb")
    parser.add_argument('--model', default="ResNet18", type=str, help="ResNet18, WideResNet-28-10")
    parser.add_argument('--beta1', default=0.9, type=float, help="effective momentum value")
    parser.add_argument('--gamma', default=0.9, type=float, help="decay rate of beta")
    parser.add_argument('--repeat', default=0, type=int)
    
    args = parser.parse_args()
    wandb_project_name = "new-sigma-CIFAR"
    wandb_exp_name = f"{args.method},{args.optimizer},b={args.batchsize},lr={args.lr},beta={args.beta1}"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,
               entity = "XXXXXX")
    wandb.init(settings=wandb.Settings(start_method='fork'))

    print('==> Preparing data..')
    if args.dataset == "CIFAR100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(15),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif args.dataset == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    device = 'cuda:0'
    if args.model == "ResNet18":
        net = resnet18()
    if args.model == "WideResNet-28-10":
        net = WideResNet28_10()
    net = net.to(device)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0)
    elif args.optimizer == "shb":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1)
        if args.model == "ResNet18":
            C = 25.318
            K = 1.77
        elif args.model == "WideResNet-28-10":
            C = 0.79
            K = 1.66
    elif args.optimizer == "nshb":
        optimizer = NSHB(net.parameters(), lr=args.lr, momentum=args.beta1)
        if args.model == "ResNet18":
            C = 128
            K = 4.5
        elif args.model == "WideResNet-28-10":
            C = 1
            K = 4.262

    if args.method == "lr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.707106781186548)
    elif args.method == "hybrid":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.866025403784439)
        increase = 1.5
    elif args.method == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif args.method == "poly":
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=200, power=args.power)
    elif args.method == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.power)
    elif args.method == "batch":
        increase = 2
    print(optimizer)

    start_epoch = 0
    next_lr = args.lr
    next_batch = args.batchsize
    next_beta = args.beta1

    if args.batchsize == 16: balance = 1.503
    elif args.batchsize == 32: balance = 1.335
    elif args.batchsize == 64: balance = 1.169
    elif args.batchsize == 128: balance = 1.0
    elif args.batchsize == 256: balance = 0.5
    elif args.batchsize == 512: balance = 0.666666666666667
    elif args.batchsize == 1024: balance = 0.5
    
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        if args.method in ["lr", "hybrid", "cosine", "poly", "exp"]:
            scheduler.step()
        elif args.method in ["batch", "hybrid"]:
            wandb.log({'batch': next_batch})
            if epoch % args.decay_epoch == 0 and epoch != 0:
                next_batch = int(next_batch * increase)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=next_batch, shuffle=True, num_workers=2)
        elif args.method in ["batch-poly"]:
            wandb.log({'batch': next_batch})
            next_batch = batch_poly(next_batch, args.epochs, epoch, args.power, balance=1.0)
        elif args.method in ["beta-step", "beta-poly", "beta-batch-poly", "beta-lr-poly"]:
            wandb.log({'beta1': next_beta})
            if args.method in ["beta-step"]:
                if epoch % args.decay_epoch == 0 and epoch != 0:
                    next_beta = next_beta * args.gamma
            elif args.method in ["beta-poly", "beta-batch-poly", "beta-lr-poly"]:
                old_batch = next_batch
                old_lr = next_lr
                if args.method in ["beta-batch-poly"]:
                    wandb.log({'batch': int(next_batch)})
                    next_batch = batch_poly(next_batch, args.epochs, epoch, args.power, balance=2.0)
                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(next_batch), shuffle=True, num_workers=2)
                elif args.method in ["beta-lr-poly"]:
                    wandb.log({'last_lr': next_lr})
                    next_lr = lr_poly(next_lr, args.epochs, epoch, args.power)
                if args.optimizer == "shb":
                    top = ((next_gamma(args.power, args.epochs, epoch) * next_delta_SHB(old_lr, old_batch, next_beta, C, K)) ** 2) - ((next_lr ** 2) * (C / next_batch))
                    bottom = (next_lr ** 2) * ((K ** 2) + (C / next_batch))
                    maru = top / bottom
                    next_beta = solve_3eq(maru, next_beta)
                    optimizer = optim.SGD(net.parameters(), lr=next_lr, momentum=next_beta)
                elif args.optimizer == "nshb":
                    top = (next_lr**2) * (1 - next_beta) * old_batch
                    bottom = (next_gamma(args.power, args.epochs, epoch)**2) * (old_lr**2) * (next_batch)
                    ato = top / bottom
                    next_beta = 1 - ato
                    optimizer = optim.SGD(net.parameters(), lr=next_lr, momentum=next_beta)
        elif args.method in ["lr-batch-poly"]:
            old_batch = next_batch
            old_lr = next_lr
            wandb.log({'last_lr': next_lr})
            wandb.log({'batch': int(next_batch)})
            next_batch = batch_poly(next_batch, args.epochs, epoch, args.power, balance)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(next_batch), shuffle=True, num_workers=2)
            if args.optimizer == "shb":
                top = next_gamma(args.power, args.epochs, epoch) * next_delta_SHB(old_lr, old_batch, next_beta, C, K)
                bottom = next_delta_SHB(1, next_batch, next_beta, C, K)
                next_lr = top / bottom
                optimizer = optim.SGD(net.parameters(), lr=next_lr, momentum=next_beta)
            elif args.optimizer == "nshb":
                next_lr = next_gamma(args.power, args.epochs, epoch) * old_lr * np.sqrt(next_batch / old_batch)
                optimizer = optim.SGD(net.parameters(), lr=next_lr, momentum=next_beta)