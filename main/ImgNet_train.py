import sys
sys.path.append("..")
import argparse
from lib.core.function import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet10i',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet10i)')
parser.add_argument('--dataset', default='ImageNet', help='dataset setting')
parser.add_argument('--COLOR_SPACE', default='RGB', help='dataset setting')
parser.add_argument('--TRAIN_JSON', default='../data/ImageNet_LT/ImageNet_LT_train.json', help='train dataset')
parser.add_argument('--VALID_JSON', default='../data/ImageNet_LT/ImageNet_LT_val.json', help='valid setting')
parser.add_argument('--loss_type', default="BaMix", type=str, help='loss type')
parser.add_argument('--train_rule', default='DRW', type=str, help='data sampling strategy for train loader')
parser.add_argument('--drw_epoch', default=150, type=int, metavar='N', help='defer re-weighting epoch')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('--no_AA', action='store_true', default=False, help='not use Auto-augment')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=180, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--startValid', default=0, type=int, metavar='N', help='the first epoch to valid')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate(default: 0.1)', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-ss', '--save-start', default=0, type=int,
                    metavar='N', help='save start epoch (default: 0)')
parser.add_argument('-s', '--save-freq', default=30, type=int,
                    metavar='N', help='save frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--devices', default='0', help='cuda device, i.e. 0 or 0,1,2,3')
parser.add_argument('--root_log',type=str, default='../log')
parser.add_argument('--root_model', type=str, default='../checkpoint')
parser.add_argument('--mixup', default=1, type=int)
parser.add_argument('--al', default=1.0, type=float)
parser.add_argument('--mixepoch', default=160, type=int)
parser.add_argument('--max_m', default=0.5, type=float)
parser.add_argument('--be', default=0.9999, type=float)
parser.add_argument('--hp', default=1.0, type=float)
def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule,
                                args.exp_str,'alpha', str(args.al), 'max_m',str(args.max_m), 'mixepoch',str(args.mixepoch), str(args.hp)])
    main_worker(args)


if __name__ == '__main__':
    main()
