# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:
import os
# base_path = './data'
# # a = ''
# # with open(os.path.join(base_path, '40_garbage-classify-for-pytorch/garbage_class.txt'), 'r', encoding='UTF-8') as f:
# #     lines = f.readlines()
# #     for line in lines:
# #         a += line.strip().replace(' ', ':').replace(',','') + '\n'
# #
# # print('')
# # with open(os.path.join(base_path, '40_garbage-classify-for-pytorch/garbage_label.txt'), 'w', encoding='UTF-8') as f:
# #     f.write(a)
import codecs
# class_id2name = {}
# for line in codecs.open('./data/40_garbage-classify-for-pytorch/garbage_label.txt', 'r'):
#     line = line.strip()
#     _id = line.split(":")[0]
#     _name = line.split(":")[1]
# #     class_id2name[int(_id)] = _name
import glob
import shutil
base_dir = r'D:\College\CS513\garbage_classfication\data\40_garbage-classify-for-pytorch\val'
dst_dir = r'D:\College\CS513\garbage_classfication\data\40_garbage-classify-for-pytorch\test'
A = glob.glob(base_dir + '\\*')
for clas in A:
    n = glob.glob(clas + '\\*')
    i=0
    for img in n:
        if i<3:
            shutil.copy(img, dst_dir)
            i+=1
        else:
            break

#
# import glob
# import shutil
# base_dir = r'D:\College\CS513\garbage_classfication\data\40_garbage-classify-for-pytorch\train'
# A = glob.glob(base_dir + '\\*')
# base_path = './data'
# i = 0
# with open(os.path.join(base_path, '40_garbage-classify-for-pytorch/garbage_target.txt'), 'w') as f:
#     for clas in A:
#         name = clas.split('\\')[-1]
#         f.write("{}:{}\n".format(i, name))
#         i+=1

