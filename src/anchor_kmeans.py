# ref:https://blog.csdn.net/m_buddy/article/details/82926024
# -*- coding=utf-8 -*-
from glob import glob

import os
import sys
import numpy as np

from kmeans import kmeans, avg_iou

# path which includes both images and label folder
ROOT_PATH = '/home/shaoheng/Documents/cars_label_FRNet/cars'
# class amount
CLUSTERS = 2
# input size of the model
SIZE = 640


# load YOLO-format training set
def load_dataset(path):
    jpegimages = os.path.join(path, 'foryolo_origin')
    if not os.path.exists(jpegimages):
        print('no JPEGImages folders, program abort')
        sys.exit(0)
    labels_txt = os.path.join(path, 'foryolo_origin')
    if not os.path.exists(labels_txt):
        print('no labels folders, program abort')
        sys.exit(0)

    # load all the txt file except the classed.txt, test.txt, train.txt
    label_file = glob('%s/[!classes, test, train]*.txt' % labels_txt)
    print('label count: {}'.format(len(label_file)))
    dataset = []

    for label in label_file:
        with open(os.path.join(labels_txt, label), 'r') as f:
            txt_content = f.readlines()

        for line in txt_content:
            line_split = line.split(' ')
            roi_width = float(line_split[-2])
            roi_height = float(line_split[-1])
            if roi_width == 0 or roi_height == 0:
                continue
            dataset.append([roi_width, roi_height])
            # print([roi_with, roi_height])

    return np.array(dataset)


data = load_dataset(ROOT_PATH)
out = kmeans(data, k=CLUSTERS)

print(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}-{}".format(out[:, 0] * SIZE, out[:, 1] * SIZE))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))