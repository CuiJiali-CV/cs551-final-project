# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data: 
import codecs
import os
import torch

import torchvision.transforms as transforms

import base64
from torchvision import datasets
def class_id2name():
    '''
    标签关系映射
    :return:
    '''

    clz_id2name = {}
    path = './data/40_garbage-classify-for-pytorch'
    for line in codecs.open(path + '/garbage_label.txt','r',encoding='utf-8'):
        line = line.strip()
        _id = line.split(":")[0]
        _name = line.split(":")[1]
        clz_id2name[int(_id)] = _name
    return clz_id2name

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    # 保存断点信息
    filepath = os.path.join(checkpoint, filename)
    print('checkpoint filepath = ',filepath)
    torch.save(state, filepath)
    # 模型保存
    if is_best:
        model_name = 'garbage_resnext101_model_' + str(state['epoch']) + '_' + str(
            int(state['train_acc'] * 100))+ '_' + str(
            int(state['test_acc'] * 100)) + '.pth'
        print('Validation loss decreased  Saving model ..,model_name = ', model_name)
        model_path = os.path.join(checkpoint, model_name)
        print('model_path = ',model_path)
        # state_dict = {
        #     'clf': state['state_dict'],
        #     'epoch': state['epoch'],
        #     'opt': state['optimizer']
        # }
        torch.save(state['state_dict'], model_path)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

def transform_image(img_bytes, i):
    b64image = base64.b64encode(img_bytes.read())

    with open("./test/cache/image{}.jpg".format(i), "wb") as f:
        f.write(base64.b64decode(b64image))


def get_dataloader(batch_size):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_data = datasets.ImageFolder(root='./test', transform=preprocess)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False)
    return test_loader