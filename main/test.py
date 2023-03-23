import sys
sys.path.append("..")
import argparse
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from lib import models
from lib.utils.utils import *
from lib.dataset.ImageNet import ImageNet
import numpy as np
import os
import tqdm
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch Cifar Training')

parser.add_argument('--resume', default='',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--COLOR_SPACE', default='RGB', help='dataset setting')
parser.add_argument('--VALID_JSON', default='../data/ImageNet_LT/ImageNet_LT_val.json', help='valid setting')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--root_log', type=str, default='../log')
parser.add_argument('--devices', default='0', help='cuda device, i.e. 0 or 0,1,2,3')


def main():
    args = parser.parse_args()
    # Identify store_name
    args.store_name = args.resume.split("/")[-2]
    name_list = args.store_name.split("_")
    args.dataset = name_list[0]
    args.arch = name_list[1]
    if name_list[2] == 'LDAM' :
        use_norm = True
        print("LDAM uses NormedLinear")
    elif name_list[2] == 'BaMix':
        use_norm = True
        print("BaMix uses NormedLinear")
    else:
        use_norm = False

    # prepare dataset and model architecture
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == 'cifar10':
            num_classes = 10
            val_dataset = datasets.CIFAR10(root='../../dataset/imbalance_cifar', train=False, download=True,
                                           transform=transform_val)
        else:
            num_classes = 100
            val_dataset = datasets.CIFAR100(root='../../dataset/imbalance_cifar', train=False, download=True,
                                            transform=transform_val)
    elif args.dataset == 'ImageNet':
        num_classes = 1000
        transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_dataset = ImageNet("valid", args=args, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    # prepare_gpus
    device_ids = []
    if args.devices:
        device_list = args.devices.split(',')
        for i in range(len(device_list)):
            device_ids.append(int(device_list[i]))
    else:
        for i in range(torch.cuda.device_count()):
            device_ids.append(i)
    print("Use GPU: {} for testing".format(device_ids))
    args.gpu = device_ids[0]
    torch.cuda.set_device(args.gpu)

    # resume model
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        ckp = list(checkpoint['state_dict'].items())
        if str(ckp[-1][0]).startswith("module."):
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = model.cuda(args.gpu)
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit()

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    top1=top1, top5=top5))
            print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        flag = 'val'
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(flag=flag, top1=top1, top5=top5))
        out_cls_acc = '%s Class Accuracy: %s' % (
            flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)


if __name__ == '__main__':
    main()

