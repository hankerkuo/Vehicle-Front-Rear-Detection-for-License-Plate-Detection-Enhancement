# this script is used to generate image augmentation
from glob import glob
from os.path import splitext, join, basename
from imgaug import augmenters as iaa

import cv2
import numpy as np
import imgaug as ia

ia.seed(1)

'''
Custom Parameters
'''
images_paths = glob('/home/shaoheng/Documents/cars_label_FRNet/cars/foryolo/*.jpg')
out_dir = '/home/shaoheng/Documents/cars_label_FRNet/cars/foryolo_aug'
augtype = '_fliprot15'  # used for the name for images after augmentation

images = []
images_bbs = []

for image_path in images_paths:
	image = cv2.imread(image_path)
	images.append(image)

for i, image in enumerate(images):
	width = image.shape[1]
	height = image.shape[0]
	# image = ia.imresize_single_image(image, (298, 447))

	# read for the bounding box information for each image
	bb_lst = []
	with open(splitext(images_paths[i])[0] + '.txt', 'r') as f:
		for line in f.readlines():
			line = line.split(' ')
			x1 = (float(line[1]) - float(line[3]) / 2)
			x2 = (float(line[1]) + float(line[3]) / 2)
			y1 = (float(line[2]) - float(line[4][:-1]) / 2)
			y2 = (float(line[2]) + float(line[4][:-1]) / 2)
			bb_lst.append(ia.BoundingBox(x1=x1*width, x2=x2*width, y1=y1*height, y2=y2*height))

	bbs = ia.BoundingBoxesOnImage(bb_lst, shape=image.shape)
	images_bbs.append(bbs)


seq = iaa.Sequential([
	# iaa.GammaContrast(1.5),
	# iaa.Affine(translate_percent={"x": 0.1}, scale=0.8),
	# iaa.Affine(rotate=15),
	# iaa.CropAndPad(px=(-100, 0), sample_independently=False)
	# iaa.AdditiveGaussianNoise(scale=0.2*255),
	# iaa.Sharpen(alpha=0.5)
	iaa.Fliplr(1),
	# iaa.Affine(rotate=-15)

])

seq_det = seq.to_deterministic()
# image_aug = seq_det.augment_images(images)
# bbs_aug = seq_det.augment_bounding_boxes(images_bbs)

for i, image_path in enumerate(images_paths):

	image_aug = seq_det.augment_image(images[i])
	bbs_aug = seq_det.augment_bounding_boxes(images_bbs)[i]
	bbs = bbs_aug.to_xyxy_array()

	width = image_aug.shape[1]
	height = image_aug.shape[0]

	cv2.imwrite(join(out_dir, basename(splitext(images_paths[i])[0]) + augtype + '.jpg'), image_aug)

	out_txt = open(join(out_dir, basename(splitext(images_paths[i])[0]) + augtype + '.txt'), 'w')

	with open(splitext(images_paths[i])[0] + '.txt', 'r') as f:
		for k, line in enumerate(f.readlines()):
			line = line.split(' ')
			bb = bbs[k]
			cx = np.around((bb[2] + bb[0]) / width / 2, decimals=6)
			cy = np.around((bb[3] + bb[1]) / height / 2, decimals=6)
			w = np.around((bb[2] - bb[0]) / width, decimals=6)
			h = np.around((bb[3] - bb[1]) / height, decimals=6)
			out_txt.write(line[0]+' '+cx.astype(str)+' '+cy.astype(str)+' '+w.astype(str)+' '+h.astype(str)+'\n')

	out_txt.close()

	# ia.imshow(bbs_aug.draw_on_image(image_aug, thickness=2))