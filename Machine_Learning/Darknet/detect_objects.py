import darknet as d
import cv2

def gstreamer_pipeline(
    #capture_width=1280,
    #capture_height=720,
    #display_width=1280,
    #display_height=720,
    #framerate=60,
    #flip_method=0,

    capture_width=320,
    capture_height=180,
    display_width=320,
    display_height=180,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


		

def show_camera():
	# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
	print(gstreamer_pipeline(flip_method=0))
	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

	#Intialize Net 
	net = d.load_net("yolov3-tiny.cfg".encode(), "yolov3-tiny.weights".encode(), 0)
	meta = d.load_meta("voc.data".encode())


	if cap.isOpened():
		window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
		# Window
		while cv2.getWindowProperty("CSI Camera", 0) >= 0:
			#Get the Image
			ret_val, img = cap.read()

			#Feed through a Net
			r = d.detect(net, meta,img)
			cv2.imshow("CSI Camera", img)
			# This also acts as
			break
			keyCode = cv2.waitKey(30) & 0xFF
			# Stop the program on the ESC key
			if keyCode == 27:
				break
		cap.release()
		cv2.destroyAllWindows()
	else:
		print("Unable to open camera")

if __name__ == "__main__":
	show_camera()

	#print(r)
