'''
FRD Net, the function 'detect' in darknet has been modified to be able to receive cv2.imread as an input
see darknet.py for more information
'''
from os.path import splitext, basename, isdir
from os import makedirs, remove

import sys
import cv2
import numpy as np
import traceback

from src import FRD
from src.draw_BB import draw_bb
from WPOD_src.drawing_utils import draw_losangle
from WPOD_src.keras_utils import load_model, detect_lp
from WPOD_src.label import Label, lwrite, lread, Shape
from WPOD_src.utils import crop_region, image_files_from_folder, im2single
from darknet.python.darknet import detect

import src.quadrilateral_calculation as qucal
import darknet.python.darknet as dn


if __name__ == '__main__':

	# vehicle detection
	input_dir = 'samples/overlap_case'
	output_dir = 'output'

	vehicle_threshold = .5

	vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
	vehicle_netcfg = 'data/vehicle-detector/yolo-voc.cfg'
	vehicle_dataset = 'data/vehicle-detector/voc.data'

	vehicle_net = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
	vehicle_meta = dn.load_meta(vehicle_dataset)

	imgs_paths = image_files_from_folder(input_dir)
	imgs_paths.sort()

	if not isdir(output_dir):
		makedirs(output_dir)

	print '\tSearching for vehicles using YOLO...'

	for i, img_path in enumerate(imgs_paths):

		print '\tScanning %s' % img_path
		img = cv2.imread(img_path)

		bname = basename(splitext(img_path)[0])

		R, _ = detect(vehicle_net, vehicle_meta, img, thresh=vehicle_threshold)

		R = [r for r in R if r[0] in ['car', 'bus']]

		print '\t\t%d cars found' % len(R)

		if len(R):

			WH = np.array(img.shape[1::-1], dtype=float)
			Lcars = []

			for i, r in enumerate(R):
				cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
				tl = np.array([cx - w / 2., cy - h / 2.])
				br = np.array([cx + w / 2., cy + h / 2.])
				label = Label(0, tl, br)
				Lcars.append(label)

			lwrite('%s/%s_cars.txt' % (output_dir, bname), Lcars)

	# license plate detection
	try:

		# colors are BGR in opencv
		YELLOW = (0, 255, 255)
		RED = (0, 0, 255)
		PINK = (232, 28, 232)

		input_dir = output_dir
		lp_threshold = 0.5

		wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
		wpod_net = load_model(wpod_net_path)

		print 'Searching for license plates using WPOD-NET'

		for i, img_path in enumerate(imgs_paths):

			print '\t Processing %s' % img_path
			bname = splitext(basename(img_path))[0]
			img = cv2.imread(img_path)
			label_path = '%s/%s_cars.txt' % (output_dir, bname)

			plates = []
			car_labels = lread(label_path)
			# remove the LP position information txt
			remove('%s/%s_cars.txt' % (output_dir, bname))

			for j, car_label in enumerate(car_labels):

				car = crop_region(img, car_label)
				ratio = float(max(car.shape[:2])) / min(car.shape[:2])
				side = int(ratio * 288.)
				bound_dim = min(side + (side % (2 ** 4)), 608)
				print "\t\tvehicle %d, Bound dim: %d, ratio: %f" % (j, bound_dim, ratio)

				Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(car), bound_dim, 2 ** 4, (240, 80), lp_threshold)

				if len(LlpImgs):
					Ilp = LlpImgs[0]
					Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
					Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

					s = Shape(Llp[0].pts)

					# s.pts is the points for LP, it is a numpy array with shape(2, 4)
					# this part is used to reconstruct the coordinates of LP into original image pixel scale
					# also append j into the plates_cor to record its corresponding car
					pts = s.pts * car_label.wh().reshape(2, 1) + car_label.tl().reshape(2, 1)
					ptspx = pts * np.array(img.shape[1::-1], dtype=float).reshape(2, 1)
					plates.append([j, ptspx])

					# draw_losangle(img, ptspx, RED, 3)
					# cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp * 255.)
					# writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])

			# this part is used to detect the overlapped LP
			plates_cor = [i[1] for i in plates]
			non_overlap_plates = []
			cars_processed = []
			if len(plates) > 1 and qucal.overlap(np.array(plates_cor)):

				FRD_record = open(output_dir + '/%s.txt' % bname, 'w')

				for ele in qucal.overlap(np.array(plates_cor)):

					print '\t\t\toverlapped LP found:', ele.couple()
					FRD_record.write('%s %s\n' % ('overlapped LP found:', ele.couple()))

					car_1 = plates[ele.couple()[0]][0]
					car_2 = plates[ele.couple()[1]][0]
					cars_processed.append(car_1)
					cars_processed.append(car_2)
					print '\t\t\trelated car:', car_1, 'with', car_2
					FRD_record.write('%s %d %s %d\n' % ('related car:', car_1, 'with', car_2))

					uni_area = qucal.union_area(np.array([car_labels[car_1].tl(), car_labels[car_1].br()]),
												np.array([car_labels[car_2].tl(), car_labels[car_2].br()]))
					uni_img = crop_region(img, uni_area)

					try:
						frs, cate = FRD.fr_detect(uni_img)
						fr_lst = []
						for fr in frs:
							fr_lst.append(Label(tl=fr.tl()*uni_area.wh() + uni_area.tl(),
												br=fr.br()*uni_area.wh() + uni_area.tl()))

						for k, fr in enumerate(fr_lst):
							owner_car = None
							if qucal.FRCar(fr, car_labels[car_1]).cover_rate() >= \
								qucal.FRCar(fr, car_labels[car_2]).cover_rate():

								print '\t\t\tfr:', k, 'car:', car_1, 'has better cover rate'
								FRD_record.write('%s %d %s %d %s \n' % ('fr:', k, 'car:', car_1, 'has better cover rate'))

								owner_car = car_1

								non_overlap_plates.append(ele.larger_plate)

								if qucal.overlap(np.array([ele.larger_plate, fr.quadrilateral_format() *
												 np.array(img.shape[1::-1], dtype=float).reshape(2, 1)])):
									print '\t\t\tthis plate belongs to car:', car_1
									FRD_record.write('%s %d\n' % ('this plate belongs to car:', car_1))
							else:

								print '\t\t\tfr:', k, 'car:', car_2, 'has better cover rate'
								FRD_record.write('%s %d %s %d %s \n' % ('fr:', k, 'car:', car_2, 'has better cover rate'))
								owner_car = car_2

								non_overlap_plates.append(ele.larger_plate)

								if qucal.overlap(np.array([ele.larger_plate, fr.quadrilateral_format() *
												 np.array(img.shape[1::-1], dtype=float).reshape(2, 1)])):
									print '\t\t\tthis plate belongs to car:', car_2
									FRD_record.write('%s %d\n' % ('this plate belongs to car:', car_2))

							# draw front & rear BB
							draw_bb(img, fr, cate=cate[k], index=str(owner_car), text_color=(255, 255, 255))

					except:
						traceback.print_exc()

				FRD_record.close()

			# put the other plates into the list
			for plate in plates:
				if plate[0] in cars_processed:
					continue
				else:
					non_overlap_plates.append(plate[1])

			for plate_cor in non_overlap_plates:
				# draw plates
				draw_losangle(img, plate_cor, RED, 3)

			for j, car_label in enumerate(car_labels):
				# draw car BB
				draw_bb(img, car_label, cate='car', index=str(j), bg_color=YELLOW, text_color=(0, 0, 0))

			cv2.imwrite('%s/%s_output.png' % (output_dir, bname), img)
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
