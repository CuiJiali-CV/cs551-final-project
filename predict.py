# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data: 
import codecs
from args import args
import torch
from model import initital_model, evaluate
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import os
class_id2name = {}
for line in codecs.open('./data/40_garbage-classify-for-pytorch/garbage_label.txt', 'r'):
    line = line.strip()
    _id = line.split(":")[0]
    _name = line.split(":")[1]
    class_id2name[int(_id)] = _name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(class_id2name)
model_name = args.model_name
model_path = args.resume

model_ft = initital_model(model_name, num_classes, feature_extract=True)
model_ft.to(device)
# pretrained_dict = model_ft.state_dict()
retrained_dict = torch.load(model_path, map_location='cpu')
# pretrained_dict['fc.1.weight'] = retrained_dict['fcw']
# pretrained_dict['fc.1.bias'] = retrained_dict['fcb']
# retrained_fcb_dict = torch.load(model_path, map_location='cpu')['fcb']

# model_ft.load_state_dict(retrained_param_dict)
model_ft.load_state_dict(retrained_dict)
model_ft.eval()

data_path = './data1/40_garbage-classify-for-pytorch'
test = "{}/test".format(data_path)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_data = datasets.ImageFolder(root=test, transform=preprocess)
a = test_data.imgs

batch_size = 128
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=True)

criterion = nn.CrossEntropyLoss()
print('\nEvaluate only')
os.makedirs('./test', exist_ok=True)
test_loss, test_acc, predict_all, labels_all = evaluate(test_loader, model_ft, criterion, test=True)
print('Test Loss:%.8f,Test Acc:%.2f' % (test_loss, test_acc))