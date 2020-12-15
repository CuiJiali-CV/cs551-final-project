# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data: 

import os
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from utils import class_id2name, save_checkpoint
from model import initital_model, train, evaluate
from args import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state = {k:v for k,v in args._get_kwargs()}
print('state = ',state)

data_path = './data/40_garbage-classify-for-pytorch'
TRAIN = "{}/train".format(data_path)
VALID = "{}/val".format(data_path)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_data = datasets.ImageFolder(root=TRAIN, transform=preprocess)
val_data = datasets.ImageFolder(root=VALID, transform=preprocess)

base_path = './data'
with open(os.path.join(base_path, '40_garbage-classify-for-pytorch/garbage_target.txt'), 'w') as f:
    for k, v in train_data.class_to_idx.items():
        f.write("{}:{}\n".format(v, k))
train_data.class_to_idx
batch_size = 300
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=1, shuffle=False)

class_list = [class_id2name()[i] for i in list(range(len(train_data.class_to_idx.keys())))]
# print('class_list = ',class_list)
best_acc = 0

def run(model, train_loader,val_loader):
    global best_acc

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                args.lr)

    if args.resume:
        # --resume checkpoint/checkpoint.pth.tar
        # load checkpoint
        print('Resuming from checkpoint...')
        assert os.path.isfile(args.resume),'Error: no checkpoint directory found!!'
        checkpoint = torch.load(args.resume)
        # best_acc = checkpoint['best_acc']
        state['start_epoch'] = checkpoint['epoch']
        model.load_state_dict(checkpoint['clf'])
        optimizer.load_state_dict(checkpoint['opt'])

    if args.evaluate:
        print('\nEvaluate only')
        test_loss, test_acc, predict_all,labels_all = evaluate(val_loader,model,criterion,test=True)
        print('Test Loss:%.8f,Test Acc:%.2f' %(test_loss,test_acc))

        return

    os.makedirs(args.checkpoint, exist_ok=True)

    for epoch in range(state['start_epoch'], state['epochs'] + 1):
        print('[{}/{}] Training'.format(epoch, args.epochs))
        # train
        train_loss, train_acc = train(train_loader, model, criterion,optimizer)

        # val
        test_loss, test_acc = evaluate(val_loader, model, criterion, test=None)

        print('train_loss:%f, val_loss:%f, train_acc:%f,  val_acc:%f' % (
            train_loss, test_loss, train_acc, test_acc,))

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        save_checkpoint({
            'epoch':epoch + 1,
            'state_dict': model.state_dict(),
            'train_acc':train_acc.cpu().data.item(),
            'test_acc':test_acc.cpu().data.item(),
            'best_acc':best_acc,
            'optimizer':optimizer.state_dict()

        }, is_best, checkpoint=args.checkpoint)


    print('Best acc:')
    print(best_acc)

if __name__ == '__main__':
    print("hello")

    # init model
    model_name = args.model_name
    num_classes = args.num_classes
    model_ft = initital_model(model_name, num_classes, feature_extract=True)

    model_ft.to(device)

    run(model_ft, train_loader, val_loader)