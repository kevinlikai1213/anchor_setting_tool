# -*- coding=utf-8 -*-
import glob
import os
import sys
# import xml.etree.ElementTree as ET
import json
import numpy as np
from kmeans import kmeans, avg_iou

# 根文件夹
ROOT_PATH = '/data/DataBase/YOLO_Data/V3_DATA/'
Json_path = '/home/luck/models/official/cv/yolov3_resnet18/instances_train.json'
# 聚类的数目
CLUSTERS = 18
# 模型中图像的输入尺寸，默认是一样的
SIZE = 2048

# 加载YOLO格式的标注数据
def load_dataset(path):
    jpegimages = os.path.join(path, 'JPEGImages')
    if not os.path.exists(jpegimages):
        print('no JPEGImages folders, program abort')
        sys.exit(0)
    labels_txt = os.path.join(path, 'labels')
    if not os.path.exists(labels_txt):
        print('no labels folders, program abort')
        sys.exit(0)

    label_file = os.listdir(labels_txt)
    print('label count: {}'.format(len(label_file)))
    dataset = []

    for label in label_file:
        with open(os.path.join(labels_txt, label), 'r') as f:
            txt_content = f.readlines()

        for line in txt_content:
            line_split = line.split(' ')
            roi_with = float(line_split[len(line_split)-2])
            roi_height = float(line_split[len(line_split)-1])
            if roi_with == 0 or roi_height == 0:
                continue
            dataset.append([roi_with, roi_height])
            # print([roi_with, roi_height])

    return np.array(dataset)

def laodjson(js_path = 'instances_train.json'):
    coco = json.load(open(js_path,'r'))
    annos = coco['annotations']
    anchor = []
    for it in coco['annotations']:
        box = it['bbox']
        x, y = box[2], box[3]
        anchor.append([x,y])
    return np.array(anchor)

if __name__=="__main__":
    # data = load_dataset(ROOT_PATH)
    data=laodjson()
    out = kmeans(data, k=CLUSTERS)

    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}-{}".format(out[:, 0] * SIZE, out[:, 1] * SIZE))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
