
# this code was implemented based on the original contrastive learning framework by: Chen T, #Kornblith S, Norouzi M, et al. A simple framework for contrastive learning of visual #representations[C]//International conference on machine learning. PMLR, 2020: 1597-1607. 


from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization default =0.05for supCL
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    #IPS: m = 0.5756 s = 0.1435
    #MFG: m = 0.5983 s = 0.1381
    parser.add_argument('--mean', type=str, default='(0.5756,)', help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default='(0.1435,)', help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='./path', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=16, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature defalut = 0.07for supcl
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')
    parser.add_argument('--ROI', type=str, default='IPS_MFG',
                        help='ROI: IPS, MFG, IPS_MFG') # you could choose which region from (IPS,MFG and both)
    parser.add_argument('--td_param', type=float, default=1.0,
                        help='trade-off between IPS and MFG')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.ROI == 'IPS':
        opt.data_folder = opt.data_folder +'_ips'
    elif opt.ROI == 'MFG':
        opt.data_folder = opt.data_folder + '_mfg'
    elif opt.ROI == 'IPS_MFG':
        opt.data_folder = './path_ips_mfg'
    else:
        opt.data_folder = opt.dataset

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.data_folder[2:], opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        #IPS: m = 0.5756 s = 0.1435
        #MFG: m = 0.5983 s = 0.1381
        if opt.ROI == 'IPS':
            mean, std = 0.5756, 0.1435
        elif opt.ROI == 'MFG':
            mean, std = 0.5983, 0.1381
        else:
            mean = eval(opt.mean)
            std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        if opt.ROI == 'IPS_MFG':
            train_transform1 = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(0.5756, 0.1435),
            ])
            train_transform2 = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(0.5983, 0.1381),
            ])
            train_dataset1 = datasets.ImageFolder(root='./path_ips',
                                                  transform=TwoCropTransform(train_transform1))
            train_dataset2 = datasets.ImageFolder(root='./path_mfg',
                                                  transform=TwoCropTransform(train_transform2))
        else:
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)


    train_sampler = None
    if opt.ROI == 'IPS_MFG':
        train_loader1 = torch.utils.data.DataLoader(
            train_dataset1, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        train_loader2 = torch.utils.data.DataLoader(
            train_dataset2, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        return train_loader1, train_loader2
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model,roi=opt.ROI)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt,train_loader2=None):
    """one epoch training"""
    if opt.ROI == 'IPS_MFG':
        train_loader = zip(train_loader,train_loader2)
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if opt.ROI == 'IPS_MFG':
            images1 = torch.cat([images[0][0],images[0][1]],dim=0)
            labels1 = images[1]
            images2 = torch.cat([labels[0][0],labels[0][1]],dim=0)
            labels2 = labels[1]
            if torch.sum(labels1 - labels2) == 0:
                images = torch.cat([images1,images2],dim=1)
                labels = labels1
            else:
                print('Two dataloader has different order!')
                return 0
        else:
            images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        if opt.ROI == 'IPS_MFG':
            len_train_loader = len(train_loader2)
        else:
            len_train_loader = len(train_loader)
        warmup_learning_rate(opt, epoch, idx, len_train_loader, optimizer)

        # compute loss
        features = model(images,opt.td_param)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len_train_loader, batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # print info
    print('Running on: --->>>',opt.ROI)
    # build data loader
    if opt.ROI == 'IPS_MFG':
        train_loader1, train_loader2 = set_loader(opt)
    else:
        train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        if opt.ROI == 'IPS_MFG':
            loss = train(train_loader1, model, criterion, optimizer, epoch, opt,train_loader2=train_loader2)
        else:
            loss = train(train_loader, model, criterion, optimizer, epoch, opt)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()