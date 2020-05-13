import darknet as d
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
args = vars(ap.parse_args())

image = cv2.imread(args["input"])
print(type(image))
net = d.load_net("yolov3-tiny.cfg".encode(), "yolov3-tiny.weights".encode(), 0)
meta = d.load_meta("coco.data".encode())
r = d.detect(net, meta, image)
objects = [i[0].decode() for i in r]
accuracy = [i[1] for i in r]
coordinates = [i[2] for i in r] #(x,y,w,h)

print(objects)
print(confidence)
print(coordinates)
