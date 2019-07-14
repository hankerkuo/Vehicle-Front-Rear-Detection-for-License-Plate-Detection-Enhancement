import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from WPOD_src.label import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from WPOD_src.utils import crop_region, image_files_from_folder
from darknet.python.darknet import detect


if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = sys.argv[2]

		vehicle_threshold = .5

		vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = 'data/vehicle-detector/voc.data'

		vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		vehicle_meta = dn.load_meta(vehicle_dataset)

		imgs_paths = image_files_from_folder(input_dir)
		imgs_paths.sort()

		if not isdir(output_dir):
			makedirs(output_dir)

		print 'Searching for vehicles using YOLO...'

		for i,img_path in enumerate(imgs_paths):

			print '\tScanning %s' % img_path

			# extract the file name (using splitext to delete filename extension)
			bname = basename(splitext(img_path)[0])

			R,_ = detect(vehicle_net, vehicle_meta, img_path ,thresh=vehicle_threshold)

			# This part decides which categories need to be focused
			R = [r for r in R if r[0] in ['car', 'bus', 'motorbike']]
			# R = [r for r in R if r[0] in ['front', 'rear']]

			print '\t\t%d cars found' % len(R)

			if len(R):

				Iorig = cv2.imread(img_path)
				WH = np.array(Iorig.shape[1::-1],dtype=float)
				Lcars = []

				# each detected vehicle
				for i,r in enumerate(R):
					# make the bounding box coordinate be related to the original image size
					cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
					# tl-> top left, br-> bottom right
					tl = np.array([cx - w/2., cy - h/2.])
					br = np.array([cx + w/2., cy + h/2.])
					label = Label(0,tl,br)
					Icar = crop_region(Iorig,label)

					Lcars.append(label)

					cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)

				lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)