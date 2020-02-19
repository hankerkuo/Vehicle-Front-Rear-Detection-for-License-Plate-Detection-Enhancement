from os.path import splitext, basename, join

import numpy as np
import traceback
import cv2

from WPOD_src.label import Label
from WPOD_src.utils import image_files_from_folder
from src.draw_BB import draw_bb

import darknet.python.darknet as dn

# best:FRNet_YOLOv3_50000.weights & FRNet_YOLOv3_tiny_360000 (trained using k-means anchor)
FR_weights = 'data/FRD/FRNet_YOLOv3_tiny_126000.weights'
FR_netcfg = 'data/FRD/FRNet_YOLOv3_tiny.cfg'
FR_data = 'data/FRD/FRNet_YOLOv3_tiny.data'

print 'FRD Net pre-loading...'
FR_net = dn.load_net(FR_netcfg, FR_weights, 0)
FR_meta = dn.load_meta(FR_data)
threshold = 0.5


def fr_detect(img):
	print '\t\t\tdetecting front and rear using FRD..., Model:', FR_netcfg
	results, wh = dn.detect(FR_net, FR_meta, img, threshold)

	# the results will be list according to its probability , high prob -> low prob
	if len(results):
		print '\t\t\tFR detection completed'
		FRs = []
		category = []
		for i, result in enumerate(results):
			WH = np.array(img.shape[1::-1], dtype=float)
			cx, cy, w, h = (np.array(result[2]) / np.concatenate((WH, WH))).tolist()
			tl = np.array([cx - w / 2., cy - h / 2.])
			br = np.array([cx + w / 2., cy + h / 2.])
			print '\t\t\tFR number', i, 'position:', tl, br, 'prob:', result[1]
			FRs.append(Label(tl=tl, br=br))
			category.append(result[0])
		return np.array(FRs), category

	else:
		print '\t\t\tFR detection failed'