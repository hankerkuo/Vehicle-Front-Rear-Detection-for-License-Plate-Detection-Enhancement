import sys
import cv2
import numpy as np

from glob						import glob
from os.path 					import splitext, basename, isfile
from WPOD_src.utils import crop_region, image_files_from_folder
from WPOD_src.drawing_utils import draw_label, draw_losangle, write2img
from WPOD_src.label import lread, Label, readShapes

from pdb import set_trace as pause


YELLOW = (  0,255,255)
RED    = (  0,  0,255)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

img_files = image_files_from_folder(input_dir)

for img_file in img_files:

	bname = splitext(basename(img_file))[0]

	I = cv2.imread(img_file)

	detected_cars_labels = '%s/%s_cars.txt' % (output_dir,bname)

	Lcar = lread(detected_cars_labels)

	sys.stdout.write('%s' % bname)

	if Lcar:

		for i,lcar in enumerate(Lcar):

			draw_label(I,lcar,color=YELLOW,thickness=3)

			lp_label 		= '%s/%s_%dcar_lp.txt'		% (output_dir,bname,i)
			lp_label_str 	= '%s/%s_%dcar_lp_str.txt'	% (output_dir,bname,i)

			if isfile(lp_label):

				Llp_shapes = readShapes(lp_label)
				# Llp_shapes[0].pts -> the coordinates for the four points, from top left and clock wise
				# x1,x2,x3,x4,y1,y2,y3,y4
				# *important, these points are ratio w.r.t cropped image (single car image)
				# lcar.wh() -> the coordinates (x,y) for tl-br, shape is (1, 2), change to (2, 1) by .reshape(2,1)
				# so "Llp_shapes[0].pts*lcar.wh().reshape(2,1)" will convert the coordinates to the scale of whole image
				# and then "+ lcar.tl().reshape(2,1)" will obtain the final coordinates
				pts = Llp_shapes[0].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
				# why always I.shape[1::-1] because opencv uses (y, x) and here we use (x, y)
				# ptspx -> so here we convert the pts from ratio to real pixel position of the original image
				ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
				# "draw_losangle" is a function used to draw the parallelogram with given four points
				# because a rectangle after affine transformation will always be Parallelogram
				draw_losangle(I,ptspx,RED,3)

				'''Draw lp OCR'''
				'''
				if isfile(lp_label_str):
					with open(lp_label_str,'r') as f:
						lp_str = f.read().strip()
					llp = Label(0,tl=pts.min(1),br=pts.max(1))
					write2img(I,llp,lp_str)

					sys.stdout.write(',%s' % lp_str)
				'''

	cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I)
	sys.stdout.write('\n')


