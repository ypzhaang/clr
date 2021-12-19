from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, en_accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    #IPS: m = 0.5756 s = 0.1435
    #MFG: m = 0.5983 s = 0.1381
    parser.add_argument('--mean', type=str, default='(0.5756,)', help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default='(0.1435,)', help='std of dataset in path in form of str tuple')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100','path'], help='dataset')
    parser.add_argument('--size', type=int, default=16, help='parameter for RandomResizedCrop')

    # other setting
    # temperature defalut = 0.07for supcl
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--ckpt', type=str, default='./path',
                        help='path to pre-trained model')
    parser.add_argument('--ROI', type=str, default='IPS',
                        help='ROI: IPS, MFG, IPS_MFG')
    parser.add_argument('--data_folder', type=str, default='./path', help='path to custom dataset')
    parser.add_argument('--is_val', type=str, default='False', help='True false for isval')
    parser.add_argument('--td_param', type=float, default=1.0,
                        help='trade-off between IPS and MFG')
    parser.add_argument('--fea_sav', type=float, default=1.0,
                        help='1 is to save, 0 is to not save')

    opt = parser.parse_args()

    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    # set the path according to the environment
    #opt.data_folder = './path/'

    if opt.ROI == 'IPS':
        opt.data_folder = opt.data_folder +'_ips'
    elif opt.ROI == 'MFG':
        opt.data_folder = opt.data_folder + '_mfg'
    elif opt.ROI == 'IPS_MFG':
        opt.data_folder = './path_ips_mfg'
    else:
        opt.data_folder = opt.dataset


    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'. \
        format(opt.method, opt.data_folder[2:], opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    opt.ckpt = opt.model_path + '/' + opt.model_name +'/last.pth'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'path':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model,roi=opt.ROI)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    print('load model:',opt.ckpt)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt, train_loader2=None):
    """one epoch training"""
    if opt.ROI == 'IPS_MFG':
        train_loader = zip(train_loader,train_loader2)
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    la_fea = []
    la_label = []

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if opt.ROI == 'IPS_MFG':
            images1 = images[0]
            labels1 = images[1]
            images2 = labels[0]
            labels2 = labels[1]
            if torch.sum(labels1 - labels2) == 0:
                labels = labels1
                images = torch.cat([images1,images2],dim=1)
            else:
                print('Two train dataloader have different order!!')
                return 0

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        if opt.ROI == 'IPS_MFG':
            len_train_loader = len(train_loader2)
        else:
            len_train_loader = len(train_loader)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len_train_loader, optimizer)

        # compute loss
        with torch.no_grad():
            if opt.ROI == 'IPS_MFG':
                features = (opt.td_param*model.encoder(torch.unsqueeze(images[:,0,:],dim=1))
                            + (1-opt.td_param)*model.encoder2(torch.unsqueeze(images[:,1,:],dim=1)))
            else:
                features = model.encoder(images)
        if opt.fea_sav == 1.0:
            la_fea.append(features)
            la_label.append(labels)
        output = classifier(features.detach())# learning classifier
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        #acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        acc1,_,_ = accuracy(output, labels, topk=(1,)) # cacluate the prediction accuracy
        top1.update(acc1[0].item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len_train_loader, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    if opt.fea_sav == 1.0:
        la_fea = torch.cat(la_fea)
        np_la_fea = la_fea.detach().cpu().numpy()
        print(np_la_fea.shape)
        pca_fea = PCA(n_components=50).fit_transform(np_la_fea)
        tsne_fea= TSNE(n_components=2).fit_transform(pca_fea)
        la_label = torch.cat(la_label).detach().cpu().numpy()
        print(la_label.shape)
        save_name = './Res/IPS_fea_'+ str(opt.learning_rate)+'.mat'
        sio.savemat(save_name,{'fea':tsne_fea,'label':la_label})
        opt.fea_sav = 0
        print('fea was saved')
    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt, val_loader2=None):
    """validation"""
    if opt.ROI == 'IPS_MFG':
        val_loader = zip(val_loader, val_loader2)
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    pred_y_img = []
    tag_y_img = []
    prob_y_img = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            if opt.ROI == 'IPS_MFG':
                images1 = images[0]
                labels1 = images[1]
                images2 = labels[0]
                labels2 = labels[1]
                if torch.sum(labels1 - labels2) == 0:
                    labels = labels1
                    images = torch.cat([images1,images2],dim=1)
                else:
                    print('Two val_dataloader have different order!!')
                    return 0

            if torch.cuda.is_available():
                images = images.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if opt.ROI == 'IPS_MFG':
                features = (opt.td_param*model.encoder(torch.unsqueeze(images[:,0,:],dim=1))
                            + (1-opt.td_param)*model.encoder2(torch.unsqueeze(images[:,1,:],dim=1)))
            else:
                features = model.encoder(images)
            output = classifier(features.detach())
            loss = criterion(output, labels)

            if opt.ROI == 'IPS_MFG':
                len_val_loader = len(val_loader2)
            else:
                len_val_loader = len(val_loader)
            # update metric
            losses.update(loss.item(), bsz)
            acc1, pred_y, prob_y = accuracy(output, labels, topk=(1, ))
            top1.update(acc1[0].item(), bsz)
            pred_y_img.append(torch.squeeze(pred_y))
            tag_y_img.append(labels)
            prob_y_img.append(prob_y)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len_val_loader, batch_time=batch_time,
                       loss=losses, top1=top1))

    pred_y_img = torch.cat(pred_y_img)
    tag_y_img = torch.cat(tag_y_img)
    prob_y_img = torch.cat((prob_y_img))
    Acc_stu, stu_pred, stu_label, stu_prob = en_accuracy(pred_y_img,tag_y_img)
    print(' * Acc@1 {top1.avg:.3f} **** Acc@stu {Acc_stu:.3f}***'.format(top1=top1,Acc_stu=Acc_stu))
    # img_pred, img_label, stu_pred, stu_label
    img_res_matrix = torch.cat((pred_y_img.reshape(pred_y_img.shape[0],1),tag_y_img.reshape(tag_y_img.shape[0],1),
                                prob_y_img.reshape(prob_y_img.shape[0],1)),1)
    stu_res_matrix = torch.cat((stu_pred.reshape(stu_pred.shape[0],1),stu_label.reshape(stu_label.shape[0],1),
                               stu_prob.reshape(stu_prob.shape[0],1)),1)
    return losses.avg, top1.avg, Acc_stu, img_res_matrix, stu_res_matrix


def main():
    best_acc = 0
    best_acc_stu = 0
    opt = parse_option()

    # build data loader
    if opt.ROI == 'IPS_MFG':
        train_loader1, train_loader2, val_loader1, val_loader2 = set_loader(opt)
    else:
        train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)


    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        if opt.ROI == 'IPS_MFG':
            loss, acc = train(train_loader1, model, classifier, criterion,
                              optimizer, epoch, opt, train_loader2)
        else:
            loss, acc = train(train_loader, model, classifier, criterion,
                              optimizer, epoch, opt)

        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        if opt.ROI == 'IPS_MFG':
            loss, val_acc, Acc_stu, img_pl, stu_pl = validate(val_loader1, model, classifier, criterion, opt, val_loader2)
        else:
            loss, val_acc, Acc_stu, img_pl, stu_pl = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
        if Acc_stu > best_acc_stu:
            best_acc_stu = Acc_stu
            fname = './Res/'+ opt.ROI + '_' + str(opt.learning_rate) +'.mat'
            sio.savemat(fname, {"img_pl":img_pl.detach().cpu().numpy(),"stu_pl":stu_pl.detach().cpu().numpy()})
            if best_acc_stu > 87:
                break

    print('best accuracy for images: {:.2f}, best accuracy for student: {:.2f}'.format(best_acc,best_acc_stu))


if __name__ == '__main__':
    main()
