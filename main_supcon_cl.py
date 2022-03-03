from __future__ import print_function

import os
import sys
import argparse
import time
import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from logger import Logger
from sklearn.metrics import confusion_matrix
from torchvision import transforms, datasets
from get_transforms import get_transform
from util import TwoCropTransform, AverageMeter, accuracy, source_import
from util import adjust_learning_rate, cosine_adjust, warmup_learning_rate
from util import set_optimizer, save_model, GaussianBlur, get_priority
from networks.resnet_big import SupConResNet, SupConResNet_ClassCenter
from losses import SupConLoss, SupConLoss_Classcenter

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--use_loss', type=str, default='paco')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=192,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--resume_epoch', type=int, default=400)
    parser.add_argument('--adavancebalance', action='store_true')
    parser.add_argument('--adavancebalance_type', default='progressbalance',
                        choices=['progressbalance', 'classbalance', 'squarerootbalance'], type=str)
    parser.add_argument('--aug', type=str, default='randclsstack_sim',
                        help='data augmentation')
    parser.add_argument('--aug_plus', action='store_true',
                        help='use mocov2 augmentation')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_strategy', type=str, default='reduceonplate',
                        help='learning rate strategy')
    parser.add_argument('--lr_patient', type=int, default=10,
                        help='patience number before droppping lr')
    parser.add_argument('--min_lr', default=0.001, type=float, help='minimum learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--continue_training', default=False, type=bool)
    parser.add_argument('--ckpt', type=str, default='ISIC')
    parser.add_argument('--dataset', type=str, default='ISIC',
                        choices=['cifar10', 'cifar100', 'ISIC','APTSO','path'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--val_size', type=int, default=256, help='validation size')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--alpha', default=None, type=float,
                        help='contrast weight among samples')
    parser.add_argument('--beta', default=None, type=float,
                        help='contrast weight among samples')
    parser.add_argument('--adp_alpha', default=False, type=bool)
    parser.add_argument('--alpha_update', default='step', type=str)
    parser.add_argument('--alpha_decay', default=100, type=int)
    parser.add_argument('--alpha_min', default=0.01, type=float)

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--balance', action='store_true',
                        help='using data resampling')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # options for adaptive wt
    parser.add_argument('--adp_wt', default=False, type=bool,
                        help='use adaptive wt')
    parser.add_argument('--wt_alpha', default=0.1, type=float)
    parser.add_argument('--wt_beta', default=10.0, type=float)
    parser.add_argument('--wt_gamma', default=0.1, type=float)

    opt = parser.parse_args()
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/AdUni/{}_models'.format(opt.dataset)
    opt.tb_path = './save/AdUni/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.aug)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 10:
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize((opt.val_size, opt.val_size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'test'), transform=val_transform)
    train_transform = get_transform(opt)
    train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                         transform=TwoCropTransform(train_transform))
    if opt.adavancebalance:
        train_sampler = opt.sampler_dic['sampler'](train_dataset, **opt.sampler_dic['params'])
    else:
        train_sampler = None

    if opt.balance or opt.adavancebalance:
        shuffle = False
    else:
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=shuffle,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True)

    train_loader.dataset.dic = {i: 0 for i in set(train_loader.dataset.targets)}
    val_loader.dataset.dic = {i:0 for i in set(val_loader.dataset.targets)}
    train_loader.dataset.first = [0]
    opt.cls_num_list = []
    val_num_list = []
    for i in train_loader.dataset.targets:
        train_loader.dataset.dic[i] += 1
    for i in val_loader.dataset.targets:
        val_loader.dataset.dic[i] += 1
    for i, j in enumerate(train_loader.dataset.dic.values()):
        train_loader.dataset.first.append(train_loader.dataset.first[i] + j)
    for i in train_loader.dataset.dic:
        opt.cls_num_list.append(train_loader.dataset.dic[i])
    for i in val_loader.dataset.dic:
        val_num_list.append(val_loader.dataset.dic[i])
    train_loader.dataset.first.pop()
    opt.n_cls = len(train_loader.dataset.dic)

    return train_loader, val_loader

def set_model(opt):
    model = SupConResNet_ClassCenter(name=opt.model, n_class=opt.n_cls, drop=0.5)#SupConResNet_ClassCenter(name=opt.model, n_class=opt.n_cls)#
    if opt.use_loss == 'paco':
        criterion = SupConLoss_Classcenter(alpha=opt.alpha, temperature=opt.temp, num_classes=opt.n_cls)
    elif opt.use_loss == 'supcon':
        criterion = SupConLoss(temperature=opt.temp, num_classes=opt.n_cls)
    else:
        raise Exception('loss function not defined')
    print('===> class items ', opt.cls_num_list)
    criterion.cal_weight_for_classes(opt.cls_num_list)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        if opt.resume:
            ckpt = torch.load(opt.ckpt, map_location='cpu')
            state_dict = ckpt['model']
            model.load_state_dict(state_dict)
            print('model loaded from %s'%opt.ckpt)

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt, alpha, beta):
    """one epoch training"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_dic = {}
    n_batch = 0

    end = time.time()
    for idx, (images, labels, indexs) in enumerate(train_loader):
        n_batch += 1
        data_time.update(time.time() - end)
        labels = np.array(labels).astype(np.int16)
        bsz = labels.shape[0]
        if opt.balance:
            permutation = np.random.choice(range(bsz), bsz, replace=False).tolist()
            images[0] = images[0][permutation]
            images[1] = images[1][permutation]
            labels = labels[permutation]
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = torch.Tensor(labels).cuda(non_blocking=True).long()

        for l in labels:
            l_value = l.item()
            if l_value not in batch_dic:
                batch_dic[l_value] = 1
            else:
                batch_dic[l_value] += 1

        # compute loss
        features, logits = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(alpha, beta, features, labels=labels, sup_logits=logits)
        acc1 = accuracy(logits[:bsz], labels, topk=(1,))
        top1.update(acc1[0].item(), bsz)

        _, pred = torch.max(logits, 1)

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
        top1_val = top1.val
        top1_avg = top1.avg
        losses_val = losses.val
        losses_avg = losses.avg
        if (idx + 1) % opt.print_freq == 0:
            print(f'Train: [%d][%d/%d]\t'
                  'Acc@1 {%.3f} ({%.3f})\t'
                  'loss {%.3f} ({%.3f})' % (
                      epoch, idx + 1, len(train_loader), top1_val, top1_avg, losses_val, losses_avg
                  ))
            sys.stdout.flush()

        # Update priority weights if using PrioritizedSampler
        if hasattr(train_loader.sampler, 'update_weights'):
            if hasattr(train_loader.sampler, 'ptype'):
                ptype = train_loader.sampler.ptype
            else:
                ptype = 'score'
            ws = get_priority(ptype, logits[:bsz].detach(), labels)
            inlist = [indexs.cpu().numpy(), ws]
            if opt.sampler_defs['type'] == 'ClassPrioritySampler':
                inlist.append(labels.cpu().numpy())
            train_loader.sampler.update_weights(*inlist)
    for cls in batch_dic:
        batch_dic[cls] = batch_dic[cls] // n_batch
    print('Batch statistic: ', dict(sorted(batch_dic.items())))
    return losses.avg, top1.avg, train_loader


def validate(val_loader, model, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()
    confusion = torch.zeros(opt.n_cls, opt.n_cls)
    opt.total_logits = torch.empty((0, opt.n_cls)).cuda()
    opt.total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, indexs) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward
            features, logits = model(images)

            # update metric
            acc1 = accuracy(logits, labels, topk=(1,))
            top1.update(acc1[0].item(), bsz)
            confusion[labels.item(), torch.argmax(logits).item()] += 1

            opt.total_logits = torch.cat((opt.total_logits, logits))
            opt.total_labels = torch.cat((opt.total_labels, labels))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    accurate = torch.diagonal(confusion)
    gt = torch.sum(confusion, dim=1)
    predictions = torch.sum(confusion, dim=0)
    predictions += 1e-20
    precision = accurate / predictions
    recall = accurate / gt
    f1 = (2 * precision * recall) / (precision + recall + 1e-20)
    out_cls_f1 = 'Val Class F1-score: %s' % (
        np.array2string(f1.numpy(), separator=',', formatter={'float_kind': lambda x: "%.3f" % x}))

    print('average F1 score: ', torch.mean(f1).item() * 100)
    print('specific F1 score: ', out_cls_f1)
    print(confusion.type(torch.LongTensor))
    print(' * Acc@1 {%.3f}' % (top1.avg))
    return top1.avg, torch.mean(f1).item() * 100


def main():
    best_acc = 0
    best_f1 = 0
    opt = parse_option()
    logger = Logger(opt.log_dir)
    if opt.adavancebalance_type == 'progressbalance':
        opt.sampler_defs = {'alpha': 1.0, 'cycle': 0, 'decay_gap': 30, 'def_file': './samplers/ClassPrioritySampler.py',
            'epochs': opt.epochs, 'fixed_scale': 1, 'lam': None, 'manual_only': True, 'momentum': 0.0, 'nroot': None,
            'pri_mode': train, 'ptype': 'score', 'rescale': False, 'root_decay': None, 'type': 'ClassPrioritySampler'}
    elif opt.adavancebalance_type == 'classbalance':
        opt.sampler_defs = {'def_file': './samplers/ClassAwareSampler.py', 'num_samples_cls': 4, 'type': 'ClassAwareSampler'}
    elif opt.adavancebalance_type == 'squarerootbalance':
        opt.sampler_defs = {'alpha': 1.0, 'cycle': 0, 'decay_gap': 30, 'def_file': './samplers.MixedPrioritizedSampler.py',
            'epochs': opt.epochs, 'fixed_scale': 1, 'lam': 1.0, 'manual_only': True, 'nroot': 2.0, 'ptype': 'score',
            'rescale': False, 'root_decay': None, 'type': 'MixedPrioritizedSampler'}
    else:
        raise Exception('Sampling method not defined')
    opt.sampler_dic = {
                'sampler': source_import(opt.sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in opt.sampler_defs.items() \
                           if k not in ['type', 'def_file']}
            }
    # build data loader
    train_loader, val_loader = set_loader(opt)

    if opt.balance:
        train_loader.dataset.imgs = sorted(train_loader.dataset.imgs, key=lambda x: x[1])
        train_loader.dataset.targets = sorted(train_loader.dataset.targets)
        train_loader.dataset.samples = sorted(train_loader.dataset.samples, key=lambda x: x[1])
        max_item = max(train_loader.dataset.dic.values())
        n_batches = max_item * opt.n_cls // opt.batch_size
        class_items_per_batch = opt.batch_size // opt.n_cls

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    print('===> Args ', opt)
    alpha_new = opt.alpha
    beta_new = opt.beta
    ft_sim_norm = None
    if opt.resume and not opt.continue_training:
        val_acc, val_f1 = validate(val_loader, model, opt)
        print('Testing...')
        print('Testing Acc: %.3f and F1: %.3f'%(val_acc, val_f1))
    else:
        start_epoch = 0 if not opt.resume else opt.resume_epoch
        for epoch in range(start_epoch, opt.epochs + 1):
            if opt.adavancebalance:
                loader_use = train_loader
            elif opt.balance:
                loader_use = train_loader
                loader_use.dataset.samples = loader_use.dataset.imgs.copy()
                loader_use.dataset.samples = sorted(loader_use.dataset.samples, key=lambda x: x[1])
                indices = [np.random.choice(range(max_item), max_item, replace=False).tolist() for i in range(opt.n_cls)]
                indices = [[loader_use.dataset.first[i] + (k % loader_use.dataset.dic[i]) for k in j] for i, j in
                           enumerate(indices)]
                order = []
                for i in range(n_batches):
                    for j in range(opt.n_cls):
                        order.append(indices[j][i * class_items_per_batch:(i + 1) * class_items_per_batch])
                order = [x for item in order for x in item]

                loader_use.dataset.samples = (np.array(loader_use.dataset.samples)[order]).tolist()
            else:
                loader_use = train_loader

            # adjust learning rate and alpha
            adjust_learning_rate(opt, optimizer, epoch)
            if opt.adp_alpha:
                alpha_new, beta_new = cosine_adjust(alpha_new, beta_new, epoch, opt)
            else:
                alpha_new, beta_new = opt.alpha, opt.beta

            print('===> @Epoch %d Alpha=%.2f, Beta=%.2f'%(epoch, alpha_new, beta_new))
            time1 = time.time()

            # train for one epoch
            loss, acc, loader_use = train(loader_use, model, criterion, optimizer, epoch, opt, alpha_new, beta_new)

            if hasattr(loader_use.sampler, 'reset_weights'):
                loader_use.sampler.reset_weights(epoch)

            loss_info = {
                'Epoch': epoch,
                'Contras Loss': loss
            }
            logger.log_loss(loss_info)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f} lr {}'.format(
                epoch, time2 - time1, acc, optimizer.param_groups[0]['lr']))

            # eval for one epoch
            val_acc, val_f1 = validate(val_loader, model, opt)
            acc_info = {
                'Epoch':epoch,
                'Train_acc': acc,
                'Val_acc':val_acc,
                'Val_f1':val_f1
            }
            # Reset class weights for sampling if pri_mode is valid
            if hasattr(loader_use.sampler, 'reset_priority'):
                ws = get_priority(loader_use.sampler.ptype,
                                  opt.total_logits.detach(),
                                  opt.total_labels)
                loader_use.sampler.reset_priority(ws, opt.total_labels.cpu().numpy())
            logger.log_acc(acc_info)

            if val_acc > best_acc and val_f1 > best_f1:
                best_acc = val_acc
                best_f1 = val_f1
                save_file = os.path.join(
                    opt.save_folder, 'warm_ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, best_acc, best_f1, epoch, save_file)
            print('===> Best Acc@1 %.3f, Best F1 %.3f\n'%(best_acc, best_f1))

        print('best accuracy: ', best_acc)
        print('best f1: ', best_f1)


if __name__ == '__main__':
    main()