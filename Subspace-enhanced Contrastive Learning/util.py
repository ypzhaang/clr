from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #output = F.normalize(output,1,1)
        #print(output)

        prob, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred, prob

def en_accuracy(pred_y, target_y):
    n_stu = int(target_y.shape[0]/20)
    correct_stu = torch.zeros((n_stu))
    stu_pred = torch.zeros((n_stu))
    stu_label = torch.zeros((n_stu))
    stu_prob = torch.zeros((n_stu))
    with torch.no_grad():
        for i in range(n_stu):
            pred_y_stui = pred_y[(20*(i)):(20*(i+1))]
            tag_y_stui = target_y[(20*(i)):(20*(i+1))]
            c_r = pred_y_stui == tag_y_stui
            correct_stu[i] = sum(c_r) > 10
            stu_pred[i] = sum(pred_y_stui) > 10 # true = 1 flase =0
            stu_label[i] = sum(tag_y_stui) > 10 # true = 1 flase =0
            stu_prob[i] = sum(pred_y_stui)/20.0

        return sum(correct_stu)*100.0 / n_stu, stu_pred, stu_label, stu_prob


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
