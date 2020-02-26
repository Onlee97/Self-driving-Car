import darknet as d

net = d.load_net("yolov3-tiny.cfg".encode(), "yolov3-tiny.weights".encode(), 0)
meta = d.load_meta("voc.data".encode())
r = d.detect(net, meta, "crossingguard.jpg".encode())

print(r)