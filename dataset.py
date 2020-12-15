# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data: 

import os

from glob import glob

data_path = 'D:/College/CS513/FinalProjects/garbage_classify/garbage_classify/train_data'

# for (dirpath, dirname, filenames) in walk(data_path):
# #     print('*' * 60)
# #
# #     print('Directory path:', dirpath)
# #     print('total examples = ', len(filenames))
# #     print('file name example:', filenames[:5])


def get_image_info():
    data_path_txt = os.path.join(data_path, '*.txt')
    txt_file_list = glob(data_path_txt)

    # print(txt_file_list[:2])

    img_path_list = []
    img_name2label_dict = {}

    img_label_dict = {}  # <img_label,img_count>

    for file_path in txt_file_list:


        with open(file_path, 'r') as f:
            line = f.readline()

        # print(line)
        line = line.strip()
        img_name = line.split(',')[0]
        img_label = line.split(',')[1]
        img_label = int(img_label)

        img_name_path = os.path.join(data_path, '{}'.format(img_name))

        img_path_list.append({'img_name_path': img_name_path, 'img_label': img_label})


        img_name2label_dict[img_name] = img_label


        img_label_count = img_label_dict.get(img_label, 0)
        if img_label_count:
            img_label_dict[img_label] = img_label_count + 1
        else:
            img_label_dict[img_label] = 1

    return img_path_list, img_label_dict, img_name2label_dict

img_path_list,img_label_dict,img_name2label_dict = get_image_info()
print('img_path_list = ',img_path_list[:3])
print('img_label_dict = ',img_label_dict)
print('img_label_dict len = ',len(img_label_dict))

import random
random.shuffle(img_path_list)

img_count = len(img_path_list)
train_size = int(img_count*0.8)

print('img_count = ',img_count)
print('train_size = ',train_size)
train_img_list = img_path_list[:train_size]
val_img_list = img_path_list[train_size:]

print('train_img_list size = ',len(train_img_list))
print('val_img_list size = ',len(val_img_list))

import shutil
base_path = './data'
with open(os.path.join(base_path, '40_garbage-classify-for-pytorch/train.txt'), 'w') as f:
    for img_dict in train_img_list:


        img_name_path = img_dict['img_name_path']  # '../data/garbage_classify/train_data/img_1.jpg'
        img_label = img_dict['img_label']

        f.write("{}\t{}\n".format(img_name_path, img_label))


        garbage_classify_dir = os.path.join(base_path, '40_garbage-classify-for-pytorch/train/{}'.format(img_label))


        if not os.path.exists(garbage_classify_dir):
            os.makedirs(garbage_classify_dir)

        shutil.copy(img_name_path, garbage_classify_dir)

with open(os.path.join(base_path, '40_garbage-classify-for-pytorch/val.txt'), 'w') as f:
    for img_dict in val_img_list:

        img_name_path = img_dict['img_name_path']  # '../data/garbage_classify/train_data/img_1.jpg'
        img_label = img_dict['img_label']

        f.write("{}\t{}\n".format(img_name_path, img_label))

        # First label
        garbage_classify_dir = os.path.join(base_path, '40_garbage-classify-for-pytorch/val/{}'.format(img_label))

        ## create dir
        if not os.path.exists(garbage_classify_dir):
            os.makedirs(garbage_classify_dir)

        ## copy images
        shutil.copy(img_name_path, garbage_classify_dir)