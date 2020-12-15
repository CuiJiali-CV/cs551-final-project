# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:
import models
import torch.nn as nn
import torch
from utils import AverageMeter
from torchvision import utils as vutils
import codecs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_id2name = {}
for line in codecs.open('./data/40_garbage-classify-for-pytorch/garbage_label.txt', 'r'):
    line = line.strip()
    _id = line.split(":")[0]
    _name = line.split(":")[1]
    class_id2name[int(_id)] = _name

def set_parameter_requires_grad(model, feature_extract):

    if feature_extract:
        for param in model.parameters():
            # Freeze parameters
            param.requires_grad = False

def initital_model(model_name, num_classes, feature_extract=True):

    model_ft = None

    if model_name == 'resnext101_32x16d':
        # load facebook pre_trained_model resnext101
        model_ft = models.resnext101_32x16d_wsl()

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features
        # change fc output num
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
    elif model_name == 'resnext101_32x8d':
        # load facebook pre_trained_model resnext101
        model_ft = models.resnext101_32x8d(pretrained=True)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )

    else:
        print('Invalid model name,exiting..')
        exit()

    return model_ft

def train(train_loader, model, criterion, optimizer):
    model.train()

    losses = AverageMeter()
    acces = AverageMeter()

    for batch_index, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, y_pred = torch.max(outputs, 1)
        acc = (y_pred == targets).sum().float() / targets.shape[0]
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc, inputs.size(0))
        print('batch {} of {}, loss {}, acc {}'.format(batch_index, len(train_loader), loss.item(), acc))

    return (losses.avg, acces.avg)

def evaluate(val_loader, model, criterion, test = None):

    global best_acc
    model.eval()

    losses = AverageMeter()
    acces = AverageMeter()

    for batch_index, (inputs, targets) in enumerate(val_loader):
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if test == True:
                path = './test/imgs/batch{}.png'.format(batch_index)
                vutils.save_image(inputs, path, normalize=True, nrow=8)
                _, y_pred = torch.max(outputs, 1)
                for i in range(targets.size(0)):
                    t = targets.cpu().data[i].item()
                    p = y_pred.cpu().data[i].item()
                    if t != p:
                        print('target \033[1;31m {} \033[0m, '
                              'predict \033[1;31m {} \033[0m '.format(class_id2name[t], class_id2name[p]))
                    else:
                        print('target \033[1;36m {} \033[0m, '
                              'predict \033[1;36m {} \033[0m '.format(class_id2name[t], class_id2name[p]))

            loss = criterion(outputs, targets)
            _, y_pred = torch.max(outputs, 1)
            acc = (y_pred == targets).sum().float() / targets.shape[0]
            losses.update(loss.item(), inputs.size(0))
            acces.update(acc, inputs.size(0))
            print('batch {} of {}, loss {}, acc {}'.format(batch_index, len(val_loader), loss.item(), acc))


    if test:
        return (losses.avg, acces.avg)
    else:
        return (losses.avg, acces.avg)
