# Process images

Uses pjreddie's darknet.py code to run a network on a image stored in a numpy array.

## To run
- Install Darknet: https://pjreddie.com/darknet/install/
- Install the python package for OpenCV: `pip install opencv-python`
- Ensure libdarknet.so file exists in same directory as darknet.py
- Run detect_objects.py in python:
```python detect_objects.py -i crossing_guard.jpg```
script will print 3 lists: detected objects, confidence scores, coordinates
