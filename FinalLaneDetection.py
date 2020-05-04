#Author: Ethan Chan, 5/3/2020
#This program is used to test the Lane Detection Algorithm, please refer to the README for more detail

import cv2
import numpy as np
import sys
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#Values for Center Line
x = 640 #number for the center point x-axis of the frame
y = 360 #number for the center point y-axis of the frame
center_point = []
center_point.append([[x,y]]) #array to hold the center point array
filtered_x = 640 #number for take the Exponential filter

#Threshold Values for Gaussian Blur and Edge Detection
kernalSize = (5,5) #Kernal size threshold for Gaussian blur
sigma = 0 #Sigma threshold for Gaussian blur
lowerRangeED = 200 #Lower range for Edge Detection
upperRangeED = 240 #Upper range for Edge Detection

#Threshold Values for HoughLinesP Transform
rho = 2 #Distance precision in pixel
angle = np.pi / 180 #Angular precision in radian
min_threshold = 100 #Number of votes needed to be considered a line segment
minLineLength = 100 #Minimum length of the line segment in pixels
maxLineGap = 50 #Maximum in pixels that two line segments that can be separated and still be considered a single line segment

boundary = 1 / 3 #Number for what part of the frame you want to focus on

alpha = 0.8 #Alpha value for Exponential filter

#RGB values for the lines that are displayed
distance_color = (255, 0, 0) #Blue
center_color = (255, 255, 0) #Green
line_color = (0, 255, 0) #Green
point_color = (0, 0, 255) #Red
line_width = 10 #Line thickness

#Function that will take a given image: grayscale the image, apply Gaussian Blur, and apply edge detection
def detect_edges(lane_image, kernalSize, sigma, lowerRangeED, upperRangeED):
	gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray,kernalSize,sigma)
	canny = cv2.Canny(blur, lowerRangeED, upperRangeED)
	return canny

#Function that given an image will isolate a region of interest to apply the rest of the filters to
def region_of_interest(canny):
	height, width = canny.shape
	mask = np.zeros_like(canny)
	polygon = np.array([[
		(0, height*1/2),
		(width, height*1/2),
		(width, height),
		(0, height),
		]], np.int32)
	cv2.fillPoly(mask, polygon, 255)
	cropped_edges = cv2.bitwise_and(canny, mask)
	return cropped_edges

#Function that given the thresholds for HoughLinesP will aply the HoughLines Transform on an image to detect straight lines
def detect_line_segments(cropped_edges, rho, angle, min_threshold, minLineLength, maxLineGap):
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength, maxLineGap)
    return line_segments

#Function that takes many small line segments with their endpoints and combine them ino two lines
def average_slope_intercept(frame, line_segments, boundary):
    lane_lines = []
    center_line = []
    global center_point
    distance_line = []
    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary 

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    center_line.append(center_lineMake(frame))

    if len(left_fit) > 0 and len(right_fit) > 0:
    	center_point = []
    	lane_lines.append(make_points(frame, left_fit_average))
    	lane_lines.append(make_points(frame, right_fit_average))
    	center_point.append(midpoints(lane_lines))
    
    if len(left_fit) > 0 and len(right_fit) <= 0:
    	center_point = []
    	lane_lines.append(make_points(frame, left_fit_average))
    	lane_lines.append(right_line(frame, left_fit_average))
    	center_point.append(midpoints(lane_lines))
    
    if len(right_fit) > 0 and len(left_fit) <= 0:
    	center_point = []
    	lane_lines.append(make_points(frame, right_fit_average))
    	lane_lines.append(left_line(frame, right_fit_average))
    	center_point.append(midpoints(lane_lines))

    distance_line.append(distance_lineMake(frame, center_point))
    return lane_lines, center_point, center_line, distance_line

#Helper function that given the center point will create a distance line from the center point to the center frame line
def distance_lineMake(frame, center_point):
	height, width, _ = frame.shape
	y1 = center_point[0][0][1]
	y2 = center_point[0][0][1]
	x1 = center_point[0][0][0]
	x2 = int(width*1/2)
	print(x2 - x1)
	return [[x1, y1, x2, y2]]

#Helper function that given an image will produce a line on the center of the frame
def center_lineMake(frame):
	height, width, _ = frame.shape
	y1 = 0
	y2 = height
	x1 = int(width*1/2)
	x2 = int(width*1/2)
	return [[x1, y1, x2, y2]]

#Helper function that given an image will produce a left line when there is no left line in the frame
def left_line(frame, opposite_line):
	height, width, _ = frame.shape
	slope, intercept = opposite_line
	y1 = height
	y2 = int(y1 * 1 / 2)
	x1 = 0
	x2 = 0
	return [[x1, y1, x2, y2]]

#Helper function that given an image will produce a right line when there is no right line in the frame
def right_line(frame, opposite_line):
	height, width, _ = frame.shape
	slope, intercept = opposite_line
	y1 = height
	y2 = int(y1 * 1 / 2)
	x1 = width-1
	x2 = width-1
	return [[x1, y1, x2, y2]]

#Helper function that takes line slopes and intercepts and return endpoints of the line segment
def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 2)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

#Helper function that takes the endpoints of the line segments and create a midpoint
def midpoints(lane_lines):
	lane_lines = lane_lines
	Lx = lane_lines[0][0][2]
	Ly = lane_lines[0][0][3]
	Rx = lane_lines[1][0][2]
	Ry = lane_lines[1][0][3]
	Cx = int((Lx + Rx)/2)
	Cy = int((Ly + Ry)/2)
	global filtered_x
	filtered_x = exponential_filter(filtered_x, Cx, 0.8)
	Cx = int(filtered_x)
	return [[Cx, Cy]]

#Function that makes a Moving Mean Average using center point, previous point and an alpha value
def midpointAverage(center_point, previous_point, alpha):
	a = alpha
	x0 = previous_point[0][0][0]
	y0 = previous_point[0][0][1]
	midpoint = center_point[0][0][0]
	xky = center_point[0][0][1]
	ykx = a*x0+(1-a)*xkx
	yky = a*y0+(1-a)*xky
	return [[ykx, yky]]

#Function that makes an Exponential Filter given a filtered x, x, and alpha value
def exponential_filter(filtered_x, x, alpha):
	filtered_x = alpha*x + (1-alpha)*filtered_x
	return filtered_x

#Master function that runs all the other functions
def detect_lane(frame):
    edges = detect_edges(frame, kernalSize, sigma, lowerRangeED, upperRangeED)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges, rho, angle, min_threshold, minLineLength, maxLineGap)
    lane_lines, center_point, center_line, distance_line = average_slope_intercept(frame, line_segments, boundary)
    lane_lines_image = display_lines(frame, lane_lines, center_point, center_line, distance_line, distance_color, center_color, line_color, point_color, line_width)
    return lane_lines_image

#Helper function that given all the lines and points and display with with the line function
def display_lines(frame, lines, points, center_line, distance_line, distance_color, center_color, line_color, point_color, line_width):
    line_image = np.zeros_like(frame)
    for point in points:
        for Cx, Cy in point:
        	cv2.circle(line_image, (Cx, Cy), 20, point_color, line_width)
    for line in center_line:
        for x1, y1, x2, y2 in line:
        	cv2.line(line_image, (x1, y1), (x2, y2), distance_color, 10)
    for line in distance_line:
    	for x1, y1, x2, y2 in line:
    		cv2.line(line_image, (x1, y1), (x2, y2), distance_color, 10)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)
    return line_image

#Function that you run when you give it an "image" input
def image(image):
	frame = cv2.imread(image)
	lane_image = np.copy(frame)
	small = cv2.resize(lane_image, (0,0), fx = 0.4, fy = 0.4)
	edges = detect_lane(small)
	cv2.imshow('test', edges)
	cv2.waitKey(1)

#Function that you run when you give it a "video" input
def video(video):
	cap = cv2.VideoCapture(video)
	while(cap.isOpened()):
		_, frame = cap.read()
		#Below is a function to flip the video input if it's flipped
		#frame = cv2.flip(frame, 0)
		detected = detect_lane(frame)
		cv2.imshow('result', detected)
		cv2.waitKey(1)

if __name__ == "__main__":
	if len(sys.argv) > 1:
		format = sys.argv[1]
	if format == 'image':
		image(sys.argv[2])
	if format == 'video':
		video(sys.argv[2])