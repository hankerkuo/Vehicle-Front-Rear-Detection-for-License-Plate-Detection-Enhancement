from os.path import splitext, basename, join, isdir
from os import mkdir

import numpy as np
import cv2

from WPOD_src.label import Label
from WPOD_src.utils import image_files_from_folder, crop_region
import darknet.python.darknet as dn


def do_fr(imgs_paths):

	modelmame = basename(splitext(FR_netcfg)[0])
	weight = basename(splitext(FR_weights)[0])

	model_dir = join('output_txt/', modelmame + '_' + valid_dataset)
	final_dir = join(model_dir, weight)

	if not isdir(model_dir):
		mkdir(model_dir)
	if not isdir(final_dir):
		mkdir(final_dir)
	for img_path in imgs_paths:
		print 'FRD processing on', img_path
		img = cv2.imread(img_path)
		frds, wh = dn.detect(FR_net, FR_meta, img, threshold)
		txt_file = open(join(final_dir, basename(splitext(img_path)[0]) + '.txt'), 'w')
		if len(frds) == 0:
			txt_file.write('')
			continue
		for frd in frds:
				cx, cy, w, h = (np.array(frd[2])).tolist()
				tl = np.array([cx - w / 2., cy - h / 2.])
				br = np.array([cx + w / 2., cy + h / 2.])
				label_FR = Label(tl=tl, br=br)
				tl = label_FR.tl().astype(int)
				br = label_FR.br().astype(int)
				txt_file.write(frd[0] + ' ' + str('%.2f' % frd[1]) + ' ' + str(tl[0]) + ' ' + str(tl[1])
							   		  + ' ' + str(br[0]) + ' ' + str(br[1]) + '\n')
				print '\twrote result to', join(final_dir, basename(splitext(img_path)[0]) + '.txt')
		txt_file.close()


# this function is not working now
def do_fr_after_YOLO(imgs_paths):
	output_dir = 'output_txt/fr_after_YOLO_kr'
	# using yolo v2 - voc
	YOLO_weights = 'data/vehicle-detector/yolov3.weights'
	YOLO_netcfg = 'data/vehicle-detector/yolov3.cfg'
	YOLO_data = 'data/vehicle-detector/coco.data'

	print 'YOLOv3 weights pre-loading...'
	YOLO_net = dn.load_net(YOLO_netcfg, YOLO_weights, 0)
	YOLO_meta = dn.load_meta(YOLO_data)
	threshold = 0.5

	for img_path in imgs_paths:
		print 'detecting cars in', img_path
		img = cv2.imread(img_path)
		results, wh = dn.detect(YOLO_net, YOLO_meta, img, threshold)
		txt_file = open(join(output_dir, basename(splitext(img_path)[0]) + '.txt'), 'w')
		if len(results) == 0:
			txt_file.write('')
			continue
		for result in results:
			if result[0] in ['car', 'bus']:
				WH = np.array(img.shape[1::-1], dtype=float)
				cx, cy, w, h = (np.array(result[2]) / np.concatenate((WH, WH))).tolist()
				tl = np.array([cx - w / 2., cy - h / 2.])
				br = np.array([cx + w / 2., cy + h / 2.])
				label_sub = Label(tl=tl, br=br)
				sub_img = crop_region(img, label_sub)

				# sub_image FRD, only process the highest prob one
				print '\tFRD processing...'
				frd, _ = dn.detect(FR_net, FR_meta, sub_img, threshold)
				if len(frd) == 0:
					continue
				WH_sub = np.array(sub_img.shape[1::-1], dtype=float)
				cx, cy, w, h = (np.array(frd[0][2]) / np.concatenate((WH_sub, WH_sub))).tolist()
				tl = np.array([cx - w / 2., cy - h / 2.])
				br = np.array([cx + w / 2., cy + h / 2.])
				label_FR = Label(tl=tl, br=br)
				label_scale_up = Label(tl=label_sub.tl() * WH + label_FR.tl() * WH_sub,
									   br=label_sub.tl() * WH + label_FR.br() * WH_sub)
				tl = label_scale_up.tl().astype(int)
				br = label_scale_up.br().astype(int)
				txt_file.write(frd[0][0] + ' ' + str('%.2f' % frd[0][1]) + ' ' + str(tl[0]) + ' ' + str(tl[1])
							   			 + ' ' + str(br[0]) + ' ' + str(br[1]) + '\n')
				print '\twrote result to', join(output_dir, basename(splitext(img_path)[0]) + '.txt')
		txt_file.close()


# manual arguments, it will automatically generate .txt file with the format for calculating mAP in this project:
# https://github.com/Cartucho/mAP
# the default dir path for txt files -> output_txt/
FR_weights = '/home/shaoheng/Documents/darknet-Alex/backup/FRNet_YOLOv3/FRNet_YOLOv3_50000.weights'
FR_netcfg = 'data/FRD/FRNet_YOLOv3.cfg'
FR_data = 'data/FRD/FRNet_YOLOv3.data'
valid_dataset = '500'
ROOT = '/home/shaoheng/Documents/mAP/input/images-optional'  # root for validation images

print 'FRD Net pre-loading...'
FR_net = dn.load_net(FR_netcfg, FR_weights, 0)
FR_meta = dn.load_meta(FR_data)
threshold = 0.5

imgs_paths = image_files_from_folder(ROOT)
do_fr(imgs_paths)
# do_fr_after_YOLO(imgs_paths)