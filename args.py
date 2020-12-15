# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data: 
import argparse

parser = argparse.ArgumentParser(description='Pytorch garbage Training ')
parser.add_argument('--model_name', default='resnext101_32x8d', type=str,
                    choices=['resnext101_32x8d', 'resnext101_32x16d'],
                    help='model_name selected in train')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initital learning rate 1e-2,12-4,0.001')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--resume', default="", type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--resume', default="./checkpoint/garbage_resnext101_model_14_92_86.pth", type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('-c', '--checkpoint', default="checkpoint", type=str, metavar='PATH',
                    help='path to save checkpoint')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--num_classes', default=40, type=int, metavar='N', help='number of classes')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual epoch number')
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'], metavar='N',
                    help='optimizer(default adam)')
args = parser.parse_args()