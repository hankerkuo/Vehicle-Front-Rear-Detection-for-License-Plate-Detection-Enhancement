from WPOD_src.drawing_utils import draw_label, write_text

# colors are BGR in opencv
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
PINK = (232, 28, 232)
BLUE = (202, 77, 86)


def draw_bb(img, label, cate='', index='', bg_color=YELLOW, text_color=(255, 255, 255)):
	if cate == 'front':
		bg_color = PINK
	elif cate == 'rear':
		bg_color = BLUE
	# draw car BB
	thickness = int(max(img.shape) / 300)
	draw_label(img, label, color=bg_color, thickness=thickness)
	write_text(img, label, cate + index, bg_color=bg_color, txt_color=text_color)


