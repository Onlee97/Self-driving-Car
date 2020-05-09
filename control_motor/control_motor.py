from jetbot import Robot
import ipywidgets.widgets as widgets
from IPython.display import display
import traitlets

robot = Robot()
robot.left(speed=0.3)
import time
time.sleep(3)
robot.stop()
