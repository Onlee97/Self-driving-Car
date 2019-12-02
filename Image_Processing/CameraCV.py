import cv2
import numpy as np

def detect_edges(lane_image):
	gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray,(5, 5),0)
	canny = cv2.Canny(blur, 200, 240)
	return canny

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

def detect_line_segments(cropped_edges):
    rho = 1
    angle = np.pi / 180
    min_threshold = 10
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]),minLineLength=8, maxLineGap=4)
    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
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
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 2)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def detect_lane(frame):
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    return lane_lines_image
    
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

#cap = cv2.VideoCapture("realTest2.mp4")
#while(cap.isOpened()):
#	_, frame = cap.read()
#	frame = cv2.flip(frame, 0)
#	detected = detect_lane(frame)
#	#cv2.imshow('result', frame)
#	cv2.imshow('result', detected)
#	cv2.waitKey(1)


frame = cv2.imread('realTest1.jpg')
lane_image = np.copy(frame)
small = cv2.resize(lane_image, (0,0), fx = 0.4, fy = 0.4)
edges = detect_edges(small)
cv2.imshow('test', edges)
cv2.waitKey(0)
