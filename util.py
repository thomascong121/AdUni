from __future__ import print_function

import math
import random
import numpy as np
import torch
import importlib
import torch.optim as optim
import torch.nn.functional as F
from PIL import ImageFilter

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def logits2score(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score

def logits2entropy(logits):
    scores = F.softmax(logits, dim=1)
    scores = scores.cpu().numpy() + 1e-30
    ent = -scores * np.log(scores)
    ent = np.sum(ent, 1)
    return ent


def logits2CE(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy() + 1e-30
    ce = -np.log(score)
    return ce


def get_priority(ptype, logits, labels):
    if ptype == 'score':
        ws = 1 - logits2score(logits, labels)
    elif ptype == 'entropy':
        ws = logits2entropy(logits)
    elif ptype == 'CE':
        ws = logits2CE(logits, labels)

    return ws

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform[0](x), self.transform[1](x)]


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

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cosine_adjust(alpha, beta, epoch, opt):
    if opt.alpha_update == 'cosine':
        eta_min = opt.alpha_min
        alpha =  eta_min + (alpha - eta_min) * (
                1 + math.cos(math.pi * epoch / opt.alpha_decay)) / 2
    elif opt.alpha_update == 'linear':
        eta_min = opt.alpha_min
        eta_max = opt.alpha
        alpha = eta_max - (eta_max-eta_min)/opt.epochs * epoch
        beta = 1-alpha
    elif opt.alpha_update == 'step':
        alpha = alpha if epoch < opt.alpha_decay else opt.alpha_min
    return alpha, beta


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate

    if args.warm and epoch < args.warm_epochs:
        lr = args.learning_rate / args.warm_epochs * (epoch + 1)
    elif args.cosine:
        print('use cosine annealing')
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif args.lr_strategy == 'reduceonplate':
        return
    elif args.lr_strategy == 'step':
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

        print('===> Use Other Learning adjust')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # lr = args.learning_rate
    # if args.cosine:
    #     eta_min = lr * (args.lr_decay_rate ** 3)
    #     lr = eta_min + (lr - eta_min) * (
    #             1 + math.cos(math.pi * epoch / args.epochs)) / 2
    # else:
    #     steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    #     if steps > 0:
    #         lr = lr * (args.lr_decay_rate ** steps)
    # print('===> Adjusted lr ', lr)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


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



def save_model(model, optimizer, best_acc, best_f1, epoch, save_file, classifier=None):
    print('==> Saving @ %s'%save_file)
    state = {
        'best_acc': best_acc,
        'best_f1': best_f1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    if classifier is not None:
        state['classifier'] = classifier.state_dict()
    torch.save(state, save_file)
    del state

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def compute_crossentropyloss_manual(x, y0, opt):
    """
    x is the vector with shape (batch_size,C)
    y0 shape is the same (batch_size), whose entries are integers from 0 to C-1
    """
    loss = 0.
    n_batch, n_class = x.shape
    batch_dic = {i:0 for i in range(opt.n_cls)}
    for x1,y1 in zip(x,y0):
        class_index = int(y1.item())
        loss = loss + torch.log(torch.exp(x1[class_index])/(torch.exp(x1).sum()))
        batch_dic[class_index] += torch.log(torch.exp(x1[class_index])/(torch.exp(x1).sum())).item() / n_batch
    loss = - loss/n_batch
    return loss, batch_dic
