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
from lib.core.function import *
from tensorboardX import SummaryWriter
from lib.dataset.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from lib.dataset.ImageNet import ImageNet
from lib.utils.utils import *
from lib.utils.losses import LDAMLoss, BaMixLoss
from lib.utils.autoaugment import ImageNetPolicy
from itertools import combinations
import random
import math
import numpy as np
import torch.nn.functional as F

best_acc1 = 0
p = 0

def main_worker(args):
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    global best_acc1, p

    cudnn.benchmark = True

    # prepare dataset
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        transform_train = transforms.Compose([])
        transform_train.transforms.append(transforms.RandomCrop(32, padding=4))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        transform_train.transforms.append(transforms.ToTensor())
        transform_train.transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == 'cifar10':
            num_classes = 10
            train_dataset = IMBALANCECIFAR10(root='../../dataset/imbalance_cifar', imb_type=args.imb_type,
                                             imb_factor=args.imb_factor, rand_number=args.rand_number, train=True,
                                             download=True, transform=transform_train)
            val_dataset = datasets.CIFAR10(root='../../dataset/imbalance_cifar', train=False, download=True,
                                           transform=transform_val)
        elif args.dataset == 'cifar100':
            num_classes = 100
            train_dataset = IMBALANCECIFAR100(root='../../dataset/imbalance_cifar', imb_type=args.imb_type,
                                              imb_factor=args.imb_factor, rand_number=args.rand_number, train=True,
                                              download=True, transform=transform_train)
            val_dataset = datasets.CIFAR100(root='../../dataset/imbalance_cifar', train=False, download=True,
                                            transform=transform_val)
    elif args.dataset == 'ImageNet':
        num_classes = 1000
        transform_train = transforms.Compose([])
        transform_train.transforms.append(transforms.ToPILImage())
        transform_train.transforms.append(transforms.RandomResizedCrop(size=(224, 224),
                                                                       scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        if not args.no_AA:
            transform_train.transforms.append(ImageNetPolicy())
        transform_train.transforms.append(transforms.ToTensor())
        transform_train.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = ImageNet("train", args=args, transform=transform_train)
        val_dataset = ImageNet("valid", args=args, transform=transform_val)
    elif args.dataset == 'iNaturalist':
        num_classes = 8142
        transform_train = transforms.Compose([])
        transform_train.transforms.append(transforms.ToPILImage())
        transform_train.transforms.append(transforms.RandomResizedCrop(size=(224, 224),
                                                                       scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        if args.use_AA:
            transform_train.transforms.append(ImageNetPolicy())
        transform_train.transforms.append(transforms.ToTensor())
        transform_train.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = ImageNet("train", args=args, transform=transform_train)
        val_dataset = ImageNet("valid", args=args, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()

    print(f'cls num list: {cls_num_list}')

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # prepare_gpus
    device_ids = []
    if args.devices:
        device_list = args.devices.split(',')
        for i in range(len(device_list)):
            device_ids.append(int(device_list[i]))
    else:
        for i in range(torch.cuda.device_count()):
            device_ids.append(i)
    print("Use GPU: {} for training".format(device_ids))
    args.gpu = device_ids[0]
    torch.cuda.set_device(args.gpu)

    # create model
    print("=> creating model '{}'".format(args.arch))
    use_norm = True if args.loss_type == 'LDAM' or args.loss_type == 'BaMix' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    args.num_classes = num_classes
    if len(device_ids) > 1:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

    model = model.cuda(device=args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    with open(os.path.join(args.root_log, args.store_name, 'model.txt'), 'w') as f:
        f.write(str(model))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    best_epoch = 0
    p = np.array([0]*args.num_classes).repeat(args.num_classes).reshape(args.num_classes, -1)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        #print(p)
        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = args.be
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // args.drw_epoch  # 160
            if idx > 1:
                idx = 1
            betas = [0, args.be]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
            return

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=args.max_m, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'BaMix':
            criterion = BaMixLoss(cls_num_list=cls_num_list, max_m=args.max_m, s=30, p = p, hp = args.hp, weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return
        # train for one epoch
        train(train_loader, model, criterion, optimizer, per_cls_weights, epoch, cls_num_list, args, log_training,
              tf_writer)

        # evaluate on validation set
        if epoch >= args.startValid:
            acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
            output_best = 'Best Prec@1: %.3f' % (best_acc1)
            print(output_best)
            log_testing.write(output_best + '\n')
            log_testing.flush()

            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            if is_best:
                best_epoch = epoch
                filename = '%s/%s/best.pth.tar' % (args.root_model, args.store_name)
                torch.save(state, filename)

            if ((epoch + 1) % args.save_freq == 0 and (epoch + 1) >= args.save_start) or epoch == args.epochs - 1:
                filename = '%s/%s/epoch%s.pth.tar' % (args.root_model, args.store_name, str(epoch + 1))
                torch.save(state, filename)

            filename = '%s/%s/last.pth.tar' % (args.root_model, args.store_name)
            torch.save(state, filename)

        print('best epoch: {}\n'.format(best_epoch))


def train(train_loader, model, criterion, optimizer, per_cls_weights, epoch, cls_num_list, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        out_avg = torch.zeros(input.shape[0], args.num_classes).cuda(args.gpu, non_blocking=True)
        loss_avg = 0
        acc1 = 0
        out_avg, loss_avg, now_acc = mixup(input, target, model, criterion, optimizer, per_cls_weights, epoch,
                                               cls_num_list, args, log, tf_writer)
        # measure accuracy and record loss
        losses.update(loss_avg.item(), input.size(0))
        top1.update(now_acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def mixup(input, target, model, criterion, optimizer, per_cls_weights, epoch, cls_num_list, args, log,
          tf_writer):

    count = torch.bincount(target)
    avg_count = torch.mean(count.float())

    if epoch < args.mixepoch:
        output, loss, now_acc = combiner(model, criterion, optimizer, input, target, count, cls_num_list, args,
                                         normal=False)
    else:
        output, loss, now_acc = combiner(model, criterion, optimizer, input, target, count, cls_num_list, args,
                                         normal=True)
    return output, loss, now_acc


def combiner(model, criterion, optimizer, image, label, count, cls_num_list, args, normal = True):
    alpha = args.al
    if normal == True:
        l = 1.0
    else:
        l = np.random.beta(alpha, alpha)
    idx = torch.randperm(image.size(0))
    image_a, image_b = image, image[idx]
    label_a, label_b = label, label[idx]
    mixed_image = l * image_a + (1 - l) * image_b
    label_a = label_a.to(args.gpu)
    label_b = label_b.to(args.gpu)
    mixed_image = mixed_image.to(args.gpu)

    # compute loss, output
    output = model(mixed_image, feature_flag=False, train=True)
    if normal == True or args.loss_type == 'CE':
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)
    else:
        loss = l * criterion(output, label_a, l) + (1 - l) * criterion(output, label_b, 1 - l)
 
    func = torch.nn.Softmax(dim=1)
    now_result = torch.argmax(func(output), 1)
    now_acc = l * accuracy2(now_result.cpu().numpy(), label_a.cpu().numpy())[0] + (1 - l) * \
              accuracy2(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss, now_acc


def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    global p
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        p = cf / cf.sum(axis=1).reshape(args.num_classes ,-1)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        # tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        # tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg

def adjust_learning_rate(optimizer, epoch, args):
    epoch = epoch + 1
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if epoch <= 10:
            lr = args.lr * epoch / 10
        elif epoch > 360:
            lr = args.lr * 0.0001
        elif epoch > 320:
            lr = args.lr * 0.01
        else:
            lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.dataset == 'ImageNet':

        if epoch <= 10:
            lr = args.lr * epoch / 10
        elif epoch > 160:
            lr = args.lr * 0.001
        elif epoch > 120:
            lr = args.lr * 0.01
        elif epoch > 60:
            lr = args.lr * 0.1
        else:
            lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.dataset == 'iNaturalist':
        if epoch <= 20:
            lr = args.lr * epoch / 20
        elif epoch > 280:
            lr = args.lr * 0.001
        elif epoch > 240:
            lr = args.lr * 0.01
        elif epoch > 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        warnings.warn('Dataset is not listed')
        return


