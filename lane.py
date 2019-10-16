import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(gray, 50, 150)
	return canny

def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
	[(200, height), (1100, height), (550, 250)]
	])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, polygons, 255)
	mask_image = cv2.bitwise_and(image, mask)
	return mask_image	

def display_line(image, lines):
	lines_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			# draw in image, first point, second point, line color, line thickness
			cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
	return lines_image

def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*3/5)
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		#Fit a polynomial with specified degree to the coordinates
		#Return the parameter of the Polynomial
		parameters = np.polyfit((x1, x2), (y1, y2),1)
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))
	left_fit_average = np.average(left_fit, axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])

def main():
	image = cv2.imread("test_image.jpg")
	lane_image = np.copy(image)
	canny_image = Canny(lane_image)
	crop_image = region_of_interest(canny_image)
	
	# Image, P_resolution, Theta_resolution, Minimum point threshold, Place holder array, Minimum Line Lenght, Maximum Line Gap
	lines = cv2.HoughLinesP(crop_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
	average_lines = average_slope_intercept(lane_image, lines)
	

	lines_image = display_line(lane_image, average_lines)

	#First Image, multiply with weight1, Second Image, Multply with weight2, Add the combine Image with the weight3
	combo_image = cv2.addWeighted(lane_image, 0.8, lines_image, 1, 1)
	cv2.imshow('result', combo_image)
	cv2.waitKey(0)
	# plt.imshow(region_of_interest(canny))
	# plt.show()

if __name__ == "__main__":
	main()