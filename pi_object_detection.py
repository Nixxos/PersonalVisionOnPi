# USAGE
# python3 pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle
	
# import the necessary packages
from time import localtime, strftime, time
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import paths
from multiprocessing import Process
from multiprocessing import Queue
import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

def classify_frame(net, inputQueue, outputQueue):
	# keep looping
	while True:
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue, resize it, and
			# construct a blob from it
			frame = inputQueue.get()
			frame = cv2.resize(frame, (300, 300))
			blob = cv2.dnn.blobFromImage(frame, 0.007843,
				(300, 300), 127.5)

			# set the blob as input to our deep learning object
			# detector and obtain the detections
			net.setInput(blob)
			detections = net.forward()

			# write the detections to the output queue
			outputQueue.put(detections)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--cascade", required=True,
        help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
f=open("data.txt", "a")
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#load our pickles for facial rec
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,
	outputQueue,))
p.daemon = True
p.start()

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
peopleCount=0
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

initTime=int(time.time())
images=[]
#####################################################################################################################
totalLidarDetects=0
currentTime=0

# loop for 15 minutes
while int(time.time())-initTime<900: #should be 900
	#print(str(int(time.time())-initTime))
	#object Detection
	# grab the frame from the threaded video stream, resize it, and
	# grab its imensions
	lidarDetect=0 #0 for false 1 for true
	personDetect=0 #0 for false 1 for true
	freqDetect=0 #0 for false any other value for freqID
	frame = vs.read()
	orig=frame.copy()
	frame = imutils.resize(frame, width=400)
	(fH, fW) = frame.shape[:2]
	#print("obj setup complete")
	
	#Facial Detection
	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	#facialFrame = imutils.resize(orig, width=500)
	#gray = cv2.cvtColor(facialFrame, cv2.COLOR_BGR2GRAY)
	#rgb = cv2.cvtColor(facialFrame, cv2.COLOR_BGR2RGB)
	#print("fac setup complete")

	#object
	# if the input queue *is* empty, give the current frame to
	# classify
	if inputQueue.empty():
		inputQueue.put(frame)

	#object
	# if the output queue *is not* empty, grab the detections
	if not outputQueue.empty():
		detections = outputQueue.get()
	#object
	# check to see if our detectios are not None (and if so, we'll
	# draw the detections on the frame)
	#if detections is None:
		#print("no detection")
		#faceID+=1
		#if unencodedFaces:
			#print("unencoded faces")
			#os.system("python3 encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog")
	#p2=os.path.sep.join(["/home/pi/Desktop/LIDAR/currentBranch/lidarphoto/", "{}.png".format(str(total).zfill(5))])
	#os.system("<command>")
	tripBOOL=open("/home/pi/Desktop/FINALBRANCH/isTripped.txt", "r")
	if tripBOOL.read()=='1':
		lidarDetect=1
		tripBOOL.close()
	if detections is not None:
		#print("detection")
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			name=""
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence`
			# is greater than the minimum confidence
			if confidence < args["confidence"]:
				continue

			# otherwise, extract the index of the class label from
			# the `detections`, then compute the (x, y)-coordinates
			# of the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			dims = np.array([fW, fH, fW, fH])
			box = detections[0, 0, i, 3:7] * dims
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			if "person" in label:
				personDetect=1
				#if person in label->there is a person in the frame
				#if face =False ->no face
				#if name is unknown->there is a new face in the frame
				#if name is not unknown->we know who it is
########################################################################################################3
				#Facial Detection
				# convert the input frame from (1) BGR to grayscale (for face
				# detection) and (2) from BGR to RGB (for face recognition)
				#facialFrame = imutils.resize(orig, width=500)
				#gray = cv2.cvtColor(facialFrame, cv2.COLOR_BGR2GRAY)
				#rgb = cv2.cvtColor(facialFrame, cv2.COLOR_BGR2RGB)
				#rects = detector.detectMultiScale(gray, scaleFactor=1.1,
				#	minNeighbors=5, minSize=(30, 30),
				#	flags=cv2.CASCADE_SCALE_IMAGE)

				#if face seen
				#if rects:
				#	time =strftime("%H:%M:%S", localtime())
				#	path ="/home/pi/Downloads/OpenCVTest/kill/faceSeen/"+time
				#	print("path:\t"+path)
				#	p=os.path.sep.join([path,"{}.png".format(str(total).zfill(5))])
				#	print("file:\t"+p)
				#	cv2.imwrite(p,orig)
				#	total+=1
				#no face seen
				#if not rects:
				#	time =strftime("%H:%M:%S", localtime())
				#	path ="/home/pi/Downloads/OpenCVTest/kill/noFaceSeen/"+time
				#	print("path:\t"+path)
				#	p=os.path.sep.join([path,"{}.png".format(str(total).zfill(5))])
				#	print("file:\t"+p)
				#	cv2.imwrite(p,orig)
				#	total+=1
				# OpenCV returns bounding box coordinates in (x, y, w, h) order
				# but we need them in (top, right, bottom, left) order, so we
				# need to do a bit of reordering
				#boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
				# compute the facial embeddings for each face bounding box 
				#encodings = face_recognition.face_encodings(rgb, boxes)
				#names = []
				
				#for encoding in encodings:
					#print("I see a face")
					#face=True
					# attempt to match each face in the input image to our known
					# encodings
					#matches = face_recognition.compare_faces(data["encodings"],encoding)
					#name = "Unknown"
					# check to see if we have found a match

					#if True in matches:
						# find the indexes of all matched faces then initialize a
						# dictionary to count the total number of times each face
						# was matched
						#matchedIdxs = [i for (i, b) in enumerate(matches) if b]
						#counts = {}

						# loop over the matched indexes and maintain a count for
						# each recognized face face
						#for i in matchedIdxs:
							#name = data["names"][i]
							#counts[name] = counts.get(name, 0) + 1
						#name = max(counts, key=counts.get)

					#name = max(counts, key=counts.get)
					#print("Its "+name+"!")
##########################################################################################################################################################
					#time =strftime("%H:%M:%S", localtime())
					#Save known people
					#if "Unknown" not in name:
						#print(name+" seen at "+time)
						#path ="/home/pi/Downloads/OpenCVTest/pip-install-opencv/dataset/"+name
						#print("path:\t"+path)
						#p=os.path.sep.join([path,"{}.png".format(str(total).zfill(5))])
						#print("file:\t"+p)
						#cv2.imwrite(p,orig)
						#total+=1
						#f.write(name+" seen at "+time)
					
					#save unknown people
					#if "Unknown" in name:
						#print("unknown face detected")
						#unencodedFaces=True
						#path="/home/pi/Downloads/OpenCVTest/pip-install-opencv/dataset/"+str(faceID)
						#try:
						#	os.mkdir(path)
						#except:
						#	print("path exists")
						#print("path:\t"+path)
						#p=os.path.sep.join([path,"{}.png".format(str(total).zfill(5))])
						#print("file:\t"+p)
						#cv2.imwrite(p,orig)
						#total+=1
						#f.write(str(faceID)+" seen at "+time)

################################	Writing to a txt file test 1
				#peopleCount+=1
				#temp =str(peopleCount) +'\t'
				#temp = temp +strftime(" %H:%M:%S", localtime())
				#f.write(temp+'\n')


				
                                #if face =False ->no face
                                #if name is unknown->there is a new face in the frame
                                #if name is not unknown->we know who it is
				#if person in label->there is a person in the frame
				#time =strftime("%H:%M:%S", localtime())
				#if there is a person but no face
				#if not face:
				#	print("I sees you but not ur face")
					
				#	path="/home/pi/Downloads/OpenCVTest/pip-install-opencv/photodump"
				#	p=os.path.sep.join(["/home/pi/Downloads/OpenCVTest/pip-install-opencv/photodump/","{}.png".format(str(total).zfill(5))])
				#	cv2.imwrite(p,orig)
				#	total+=1
				#	f.write("person seen at "+time)
				
			if "person" not in label:
				continue

################################	Draw the pretty pictures
			#print("trying to draw")
			#cv2.rectangle(frame, (startX, startY), (endX, endY),
			#	COLORS[idx], 2)
			#y = startY - 15 if startY - 15 > 15 else startY + 15
			#cv2.putText(frame, label, (startX, y),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			
			# loop over the recognized faces
			#for ((top, right, bottom, left), name) in zip(boxes, names):
				# draw the predicted face name on the image
			#	cv2.rectangle(facialframe, (left, top), (right, bottom),
			#		(0, 255, 0), 2)
			#	y = top - 15 if top - 15 > 15 else top + 15
			#	cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			#		0.75, (0, 255, 0), 2)
	# show the output frame
	#comment this for final deploy
	#print("trying to display")
	#cv2.imshow("Frame", frame)
	#key = cv2.waitKey(1) & 0xFF
	
	#show the facial frame as such
	#cv2.imshow("Facial Frame", facialFrame)
	# if the `q` key was pressed, break from the loop
	#if key == ord("q"):
	#	break
	#print("displayed")
	# update the FPS counter
	if(lidarDetect==1 or personDetect==1 or freqDetect!=0):
		if(lidarDetect==1):
			totalLidarDetects+=1
		if(int(time.time())-currentTime>2):
			currentTime=int(time.time())
			print("\tDetected")
			path="/home/pi/Desktop/FINALBRANCH/photodump"
			p=os.path.sep.join([path,"{}.png".format(str(len(images)).zfill(5))])
			print(p)
			cv2.imwrite(p,orig)
			images.append([int(time.time()), lidarDetect, personDetect, freqDetect, 0])
			print(images[len(images)-1])
	fps.update()
#######################################################################################################
#process DATA
#Lidar
dumpWriter = open("/home/pi/Desktop/FINALBRANCH/dump.txt", "a")
outputWriter=open("/home/pi/Desktop/FINALBRANCH/output.txt", "a")
outputWriter.write("################################\nNew Data "+strftime("%H:%M:%S", localtime())+"\n")
print("DATA PROCESSING########################################")
lidarCount=0
for x in range(len(images)):
	time=images[x][0]
	#while the time increment is less than 5 seconds and the lidar is consistantly true
	while images[x][0]-5<time and images[x][1]==1:
		x+=1
		if x==len(images):
			lidarCount+=1
			break
	lidarCount+=1
print("LIDAR DONE")
dumpWriter.write("Total Lidar Detects:{}\n".format(str(totalLidarDetects)))

#Frequency
#freqCount=0
#for x in range(len(images)):
#	time=images[x][0]
	#while the time increment is less than 5 seconds and the lidar is consistantly true
#	while images[x][0]-5<time and images[x][3]==1:
#		x+=1
#	freqCount+=1

#Camera
names=[]
faceCount=0
x=0
#imagePaths = list(paths.list_images("photodump"))
try:#THIS TAKES 2x DATA COLLECTION TIME
	while int(time.time())-initTime<3600:
#	print(x)
#	print("!@#$@%^#@$%^$#@$%^&%$#@!#$%^&%$#@!#$%^&%$#@%^&%$#@!$%^$#@$%")
#for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	#print("[INFO] processing image {}/{}".format(i + 1,
	#	len(imagePaths)))
	#name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	#image = cv2.imread(imagePath)
		#outputWriter.write("Stage one")
		if images[x][1]+images[x][2]+images[x][3]>1: #True: # image[x][2]==1: #if person
			print("checkpoint 1")
			t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(images[x][0]))
			image = cv2.imread("/home/pi/Desktop/FINALBRANCH/photodump/{}.png".format(str(x).zfill(5)))
			#outputWriter.write("Stage two")
			if image is not None:
				#outputWriter.write("Stage Three")
				print("checkpoint 2")
				rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
				boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
				if boxes is not None: #save face boxes (not encoded as 128 measurements
					print("checkpoint 3")
					#outputWriter.write("Stage Four")
					print(boxes)
				#images[x][4]=1
					faceCount+=1
				#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				#rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
				#	minNeighbors=5, minSize=(30, 30),
				#	flags=cv2.CASCADE_SCALE_IMAGE)
				#boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
				#encodings = face_recognition.face_encodings(rgb, boxes)
				#print("\t"+str(encodings))
					facialFrame = imutils.resize(image, width=500)
					gray = cv2.cvtColor(facialFrame, cv2.COLOR_BGR2GRAY)
					rgb = cv2.cvtColor(facialFrame, cv2.COLOR_BGR2RGB)
				# detect faces in the grayscale frame
					rects = detector.detectMultiScale(gray, scaleFactor=1.1,
						minNeighbors=5, minSize=(30, 30),
						flags=cv2.CASCADE_SCALE_IMAGE)
				# OpenCV returns bounding box coordinates in (x, y, w, h) order
				# but we need them in (top, right, bottom, left) order, so we
				# need to do a bit of reordering
					boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
				# compute the facial embeddings for each face bounding box
					encodings = face_recognition.face_encodings(rgb, boxes)

					for encoding in encodings:
						print("encodings")
						matches = face_recognition.compare_faces(data["encodings"],encoding)
						name = "Unknown"

	# check to see if we have found a match
						if True in matches:
							matchedIdxs = [i for (i, b) in enumerate(matches) if b]
							counts = {}
							for i in matchedIdxs:
								name = data["names"][i]
								counts[name] = counts.get(name, 0) + 1
							name = max(counts, key=counts.get)
						print(name)
						names.append(name)
						dumpWriter.write(name+"\t"+t)
		x+=1
except:
	print("Done processing")
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("Lidar Count:{}".format(str(lidarCount)))
print("Total Lidar Detects:{}".format(str(totalLidarDetects)))
print("Face Count:{}".format(str(faceCount)))
outputWriter.write("Face Count:{}\n".format(str(faceCount)))
outputWriter.write("Total Lidar Detects:{}\n".format(str(totalLidarDetects)))
outputWriter.write("Lidar Count:{}\n".format(str(lidarCount)))
for x in range(len(names)):
	try:
		outputWriter.write(names[x]+"\t"+str(names.count(names[x]))+"\n")
		name = names[x]
		while name in names:
			names.remove(name)
	except:
		x=0
		continue
os.system("rm /home/pi/Desktop/FINALBRANCH/photodump/*")
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
