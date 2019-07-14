import sys
import cv2
import numpy as np
import traceback
import re

import darknet.python.darknet as dn

from os.path 				import splitext, basename
from glob					import glob
from darknet.python.darknet import detect
from WPOD_src.label import dknet_label_conversion
from WPOD_src.utils import nms


if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = input_dir

		ocr_threshold = .4

		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg  = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)

		imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

		print 'Performing OCR...'

		# --store the image name temporarily---
		last_handle = ''
		# -------------------------------------

		for i,img_path in enumerate(imgs_paths):
			print '\tScanning %s' % img_path

			bname = basename(splitext(img_path)[0])

			R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)

				L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])

				# ---special case for tw plates---
				lp_str_tw = ''
				if len(L) == 7:
					for k, ch in enumerate(lp_str):
						if k <= 2:
							lp_str_tw += ch
						else:
							if ch == 'I':
								lp_str_tw += '1'
							else:
								lp_str_tw += ch
				else:
					lp_str_tw = lp_str
				lp_str = lp_str_tw
				# ---------------------------------

				with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					f.write(lp_str + '\n')

				print '\t\tLP: %s' % lp_str
				'''
				# ---save the intermediate image---
				# find the original image
				lp_result = lp_str_tw  # change for overall or tw_case
				origin_img = re.findall('tmp/output/(.*?)_', img_path)
				if last_handle == '':
					font_pos = 100
					origin_img_path = 'samples/test/' + origin_img[0] + '.jpg'
					img_to_be_text = cv2.imread(origin_img_path)
					cv2.putText(img_to_be_text, lp_result, (100, font_pos), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
								fontScale=2, color=(0, 0, 255), thickness=2)
				elif last_handle == origin_img:
					font_pos += 40
					cv2.putText(img_to_be_text, lp_result, (100, font_pos), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
							fontScale=2, color=(0, 0, 255), thickness=2)
				else:
					cv2.imwrite('samples/results/'+last_handle[0]+'.png', img_to_be_text)
					font_pos = 100
					origin_img_path = 'samples/test/' + origin_img[0] + '.jpg'
					img_to_be_text = cv2.imread(origin_img_path)
					cv2.putText(img_to_be_text, lp_result, (100, font_pos), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
							fontScale=2, color=(0, 0, 255), thickness=2)

				last_handle = origin_img
				# ---------------------------------
				'''
			else:

				print 'No characters found'

			'''
			# ---save the intermediate image---
			if i + 1 == len(imgs_paths):
				print 'reach data end'
				cv2.imwrite('samples/results/' + origin_img[0] + '.png', img_to_be_text)
			# ---------------------------------
			'''

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
