from numpy.linalg import norm

import numpy as np

from WPOD_src.label import Label

class Area:
	def __init__(self, candidate=None, partner=None, larger_plate=None):
		self.candidate = candidate
		self.partner = partner
		self.larger_plate = larger_plate

	def candidate(self):
		return self.candidate

	def overlap_with(self):
		return self.partner

	def couple(self):
		return [self.candidate, self.partner]



# area of a triangle with three points in numpy array
# source: http://code.activestate.com/recipes/576896-3-point-area-finder
def area_triangle(a, b, c):
	return 0.5 * norm(np.cross(b - a, c - a))


def area_quadrilateral(plate_pts):
	assert (plate_pts.shape[0] == 2 and plate_pts.shape[1] == 4)
	plate_area = area_triangle(plate_pts[:, 0], plate_pts[:, 1], plate_pts[:, 2]) + \
				 area_triangle(plate_pts[:, 2], plate_pts[:, 3], plate_pts[:, 0])
	return plate_area


# check if a point is inside a quadrilateral
def is_inside(plate_pts, single_pt):
	assert (plate_pts.shape[0] == 2 and plate_pts.shape[1] == 4 and single_pt.shape[0] == 2)

	plate_area = area_quadrilateral(plate_pts)
	total_area = 0
	for i in range(4):
		total_area += area_triangle(single_pt, plate_pts[:, i], plate_pts[:, (i + 1) % 4])

	if total_area == plate_area:
		return True
	else:
		return False


# check if a point is near a quadrilateral, to avoid Floating-Point Arithmetic problem when counting area equalization
def is_near(plate_pts, single_pt):
	assert (plate_pts.shape[0] == 2 and plate_pts.shape[1] == 4 and single_pt.shape[0] == 2)

	plate_area = area_quadrilateral(plate_pts)
	total_area = 0
	for i in range(4):
		total_area += area_triangle(single_pt, plate_pts[:, i], plate_pts[:, (i + 1) % 4])

	if np.abs(total_area - plate_area) / plate_area <= 0.001:
		return True
	else:
		return False


# check if multiple quadrilaterals is overlapped with each other
# mul_plate_pts is a numpy array with shape(?, 2, 4)
# return value 'overlap' is a list of class 'Area'
def overlap(mul_plate_pts):
	assert (mul_plate_pts.shape[0] > 1 and mul_plate_pts.shape[1] == 2 and mul_plate_pts.shape[2] == 4)

	plate_num = mul_plate_pts.shape[0]
	overlap = []
	for i in range(plate_num):
		for j in range(i):
			# break down the four points to pts
			num = len(overlap)
			for pts in [mul_plate_pts[i, :, k] for k in range(4)]:
				if is_near(mul_plate_pts[j], pts):
					if area_quadrilateral(mul_plate_pts[i]) >= area_quadrilateral(mul_plate_pts[j]):
						larger_plate = mul_plate_pts[i]
					else:
						larger_plate = mul_plate_pts[j]
					overlap.append(Area(j, i, larger_plate))
					break
			if num != len(overlap):
				continue
			for pts in [mul_plate_pts[j, :, k] for k in range(4)]:
				if is_near(mul_plate_pts[i], pts):
					if area_quadrilateral(mul_plate_pts[i]) >= area_quadrilateral(mul_plate_pts[j]):
						larger_plate = mul_plate_pts[i]
					else:
						larger_plate = mul_plate_pts[j]
					overlap.append(Area(i, j, larger_plate))
					break
	if len(overlap) > 0:
		return overlap
	else:
		return False


# this class can obtain several relationship between (front or rear) and (the candidate car)
class FRCar:
	# fr -> coordinates of front or rear, car -> coordinates of the car, format: Label class
	def __init__(self, fr, car):
		self.fr = np.array([fr.tl(), fr.br()])
		self.car = np.array([car.tl(), car.br()])

	def IOU(self):
		wh1, wh2 = self.fr[1] - self.fr[0], self.car[1] - self.car[0]
		assert ((wh1 >= .0).all() and (wh2 >= .0).all())

		intersection_wh = np.maximum(np.minimum(self.fr[1], self.car[1]) - np.maximum(self.fr[0], self.car[0]), 0.)
		intersection_area = np.prod(intersection_wh)
		area1, area2 = (np.prod(wh1), np.prod(wh2))
		union_area = area1 + area2 - intersection_area
		return intersection_area / union_area

	# this is the rate -> (fr overlap with the car) / (total fr region)
	def cover_rate(self):
		fr_wh = self.fr[1] - self.fr[0]
		intersection_wh = np.maximum(np.minimum(self.fr[1], self.car[1]) - np.maximum(self.fr[0], self.car[0]), 0.)
		intersection_area = np.prod(intersection_wh)
		fr_area = np.prod(fr_wh)
		return intersection_area / fr_area

	# this is the rate -> (fr overlap with the car) / (total car region)
	def fr_car_rate(self):
		car_wh = self.car[1] - self.car[0]
		intersection_wh = np.maximum(np.minimum(self.fr[1], self.car[1]) - np.maximum(self.fr[0], self.car[0]), 0.)
		intersection_area = np.prod(intersection_wh)
		car_area = np.prod(car_wh)
		return intersection_area / car_area


# calculate the union area of two bounding box, format: Label class
def union_area(bb_1, bb_2):
	final_tl = np.minimum.reduce([bb_1[0], bb_1[1], bb_2[0], bb_2[1]])
	final_br = np.maximum.reduce([bb_1[0], bb_1[1], bb_2[0], bb_2[1]])
	return Label(tl=final_tl, br=final_br)


'''
for testing
'''
# x = np.array([[2, 4, 3, 1], [7, 7, 4, 4]])
# y = np.array([[5, 7, 6, 4], [7, 7, 4, 4]])
# y_1 = np.array([[3, 5, 4, 2], [7, 7, 4, 4]])
# z = np.array([x, y_1, y])
# for ele in overlap(z):
# 	print ele
# 	print ele.couple()

# rec_1 = Label(tl=np.array([0, 0]), br=np.array([0.75, 0.75]))
# rec_2 = Label(tl=np.array([0.25, 0.25]), br=np.array([1, 1]))
# relation = FRCar(rec_1, rec_2)
# print relation.fr_car_rate()
# print union_area(rec_1, rec_2).br()

# larger_plate = np.array( [[1456.63956665, 1855.03084308, 1841.31345854, 1442.92218212],
#  [1735.60156318, 1667.48873682, 1778.6130428,  1846.72586915]])
# fr  = np.array([[2341.77650478, 3134.6428854 , 3134.6428854,  2341.77650478],
#  [1437.83749213, 1437.83749213, 1916.24992808 ,1916.24992808]])
# print overlap(np.array([larger_plate, fr]))