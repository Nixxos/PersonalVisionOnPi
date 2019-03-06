import os
import threading
import time
def c1():
	print("CD")
	os.system("cd /home/pi/Desktop/FINALBRANCH")
def c2():
	print("arduino")
	os.system("python3 /home/pi/Desktop/FINALBRANCH/ArduinoREADV2.py")
def c3():
	print("lidar")
	os.system("python3 /home/pi/Desktop/FINALBRANCH/LIDARpy3.py")
def c4():
	print("camera")
	os.system("python3 /home/pi/Desktop/FINALBRANCH/pi_object_detection.py --prototxt /home/pi/Desktop/FINALBRANCH/MobileNetSSD_deploy.prototxt.txt --model /home/pi/Desktop/FINALBRANCH/MobileNetSSD_deploy.caffemodel --cascade /home/pi/Desktop/FINALBRANCH/haarcascade_frontalface_default.xml --encodings /home/pi/Desktop/FINALBRANCH/encodings.pickle")
t1 = threading.Thread(target=c1)
t2 = threading.Thread(target=c2)
t3 = threading.Thread(target=c3)
t4 = threading.Thread(target=c4)
time.sleep(10)
t1.start()
t2.start()
t3.start()
t4.start()
