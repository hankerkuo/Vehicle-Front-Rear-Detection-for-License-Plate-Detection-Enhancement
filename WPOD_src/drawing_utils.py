import numpy as np
import cv2


def draw_label(I,l,color=(255,0,0),thickness=1):
	wh = np.array(I.shape[1::-1]).astype(float)
	tl = tuple((l.tl()*wh).astype(int).tolist())
	br = tuple((l.br()*wh).astype(int).tolist())
	cv2.rectangle(I,tl,br,color,thickness=thickness)


def draw_losangle(I,pts,color=(1.,1.,1.),thickness=1):
	assert(pts.shape[0] == 2 and pts.shape[1] == 4)

	for i in range(4):
		pt1 = tuple(pts[:,i].astype(int).tolist())
		pt2 = tuple(pts[:,(i+1)%4].astype(int).tolist())
		cv2.line(I,pt1,pt2,color,thickness)


def write2img(Img,label,strg,txt_color=(0,0,0),bg_color=(255,255,255),font_size=1):
	wh_img = np.array(Img.shape[1::-1])

	font = cv2.FONT_HERSHEY_SIMPLEX

	wh_text,v = cv2.getTextSize(strg, font, font_size, 3)
	bl_corner = label.tl()*wh_img

	tl_corner = np.array([bl_corner[0],bl_corner[1]-wh_text[1]])/wh_img
	br_corner = np.array([bl_corner[0]+wh_text[0],bl_corner[1]])/wh_img
	bl_corner /= wh_img

	if (tl_corner < 0.).any():
		delta = 0. - np.minimum(tl_corner,0.)
	elif (br_corner > 1.).any():
		delta = 1. - np.maximum(br_corner,1.)
	else:
		delta = 0.

	tl_corner += delta
	br_corner += delta
	bl_corner += delta

	tpl = lambda x: tuple((x*wh_img).astype(int).tolist())

	cv2.rectangle(Img, tpl(tl_corner), tpl(br_corner), bg_color, -1)
	cv2.putText(Img,strg,tpl(bl_corner),font,font_size,txt_color,3)


def write_text(Img, label, strg, txt_color=(255, 255, 255), bg_color=(120, 120, 120)):
	wh_img = np.array(Img.shape[1::-1])
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_size = max(wh_img[1] / 600, 0.5)
	thickness = int(font_size * 2)
	to_imgscale = lambda x: (x * wh_img).astype(int)
	tpl = lambda x: tuple(x.tolist())
	# cv2.rectangle -> point coordinate is (x, y), coordinate system origin : top left
	cv2.rectangle(Img, tpl(to_imgscale(label.tl())),
				       tpl(to_imgscale(label.tl()) + (np.array([18 * len(strg), -30]) * font_size).astype(int)),
				       bg_color, -1)
	cv2.putText(Img, strg, tpl(to_imgscale(label.tl()) + [0, -5]), font, font_size, txt_color, thickness)


'''
if __name__ == '__main__':
	from WPOD_src.label import Label
	import numpy as np
	img = cv2.imread('/home/shaoheng/Documents/alpr-unconstrained-master/samples/FRD/IMG_8255.JPG')
	label = Label(tl=np.array([0.2, 0.2]), br=np.array([0.7, 0.7]))
	write_text(img, label, 'rear')

	cv2.imshow('write_text', img)
	cv2.waitKey()

	cv2.destroyAllWindows()
'''