# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data: 

from flask import Flask,request,jsonify
import torch
from utils import transform_image, class_id2name, get_dataloader
from model import initital_model
from collections import OrderedDict
import codecs
from args import args
import time
import torchvision.transforms as transforms
from torchvision import datasets
state = {k: v for k, v in args._get_kwargs()}
print("state = ", state)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data_path = './data/40_garbage-classify-for-pytorch'
TRAIN = "{}/val".format(data_path)
train_data = datasets.ImageFolder(root=TRAIN, transform=preprocess)

import os
base_path = './data'
with open(os.path.join(base_path, '40_garbage-classify-for-pytorch/garbage_target.txt'), 'w') as f:
    for k, v in train_data.class_to_idx.items():
        f.write("{}:{}\n".format(v, k))


class_id2name = {}
for line in codecs.open('./data/40_garbage-classify-for-pytorch/garbage_target.txt', 'r'):
    line = line.strip()
    _id = line.split(":")[0]
    _name = line.split(":")[1]
    class_id2name[int(_id)] = _name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(class_id2name)
model_name = args.model_name
model_path = args.resume

print("model_name = ", model_name)
print("model_path = ", model_path)

model_ft = initital_model(model_name, num_classes, feature_extract=True)
model_ft.to(device)
model_ft.load_state_dict(torch.load(model_path, map_location='cpu'))
model_ft.eval()

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/predict', methods=['POST'])

def predict():
    # request input
    files = request.files.getlist('file')


    i = 0
    for file in files:
        transform_image(file, i)
        i+=1

    test_dataloader = get_dataloader(len(files))

    # predict
    outputs = []
    for batch_index, (inputs, targets) in enumerate(test_dataloader):
        with torch.no_grad():
            t1 = time.time()
            inputs = inputs.cuda()
            output = model_ft.forward(inputs)
            outputs.append(output)
            consume = (time.time() - t1) * 1000
            consume = int(consume)
    outputs = torch.cat(outputs, dim=0)
    # API

    _, y_pred = torch.max(outputs, 1)
    pred_list = y_pred.cpu().numpy().tolist()
    dict_list = []
    for i in range(len(pred_list)):
        result = {'name': class_id2name[pred_list[i]], 'label': pred_list[i]}
        dict_list.append(result)

    result = OrderedDict(error=0, errmsg='success', consume=consume, data=dict_list)
    return jsonify(result)

if __name__ == '__main__':
    # curl -X POST -F file=@cat_pic.jpeg http://localhost:5000/predict
    app.run()