import time
from time import strftime, localtime



#/while True:
#LIDpath = '/home/pi/Desktop/LIDAR/currentBranch/lidardata.txt'
#lidar = open('/home/pi/Desktop/LIDAR/currentBranch/lidardata.txt','r')
#lidarDATA = lidar.read()

#title = 'LIDAR DATA\n'

#print(title)


#print(lidarDATA)

initTime=int(time.time())
tripCounter = 0

while int(time.time())-initTime<=30:
    
    with open('/home/pi/Desktop/LIDAR/currentBranch/lidardata.txt', 'rb') as fh:
        #first = next(fh).decode()

        fh.seek(-1000, 2)
        currentLIDAR = fh.readlines()[-1].decode()

    with open('/home/pi/Desktop/LIDAR/currentBranch/PIR.txt', 'rb') as fh:
        #first = next(fh).decode()

        fh.seek(-1000, 2)
        currentPIR = fh.readlines()[-1].decode()

    with open('/home/pi/Desktop/LIDAR/currentBranch/data.txt', 'rb') as fh:
        #first = next(fh).decode()

        fh.seek(-1000, 2)  #jostle these numbers around
        currentHUMANFRAME = fh.readlines()[-1].decode()
        

    print(strftime(" %H:%M:%S", localtime()))
    
    print('||| LIDAR: '+currentLIDAR + '||| PIR: '+currentPIR + '||| Humanframe: '+currentHUMANFRAME+'\n')
    #print('||| LIDAR: '+currentLIDAR + '||| Humanframe: '+currentHUMANFRAME+'\n')

    ####################################################
    #TRIP
    tripVALUE = 800
    #tripFile = open('/home/pi/Desktop/LIDAR/currentBranch/timesTripped.txt','a')
    ####################################################


    
    with open('/home/pi/Desktop/LIDAR/currentBranch/lidarDISTANCEREADING.txt', 'rb') as fh:
        #first = next(fh).decode()

        fh.seek(-1000, 2)
        currentLIDARDISTANCE = fh.readlines()[-1].decode()

    tripCountFile = open('/home/pi/Desktop/LIDAR/currentBranch/timesTripped.txt', 'a')
    tripBOOL = open('/home/pi/Desktop/LIDAR/currentBranch/isTripped.txt', 'w')

    if int(currentLIDARDISTANCE) < tripVALUE:
        
        tripBOOL.write('1')
        tripCounter += 1
        #tripString = tripCounter + '\n'
        tripCountFile.write(str(tripCounter) + '\t@\t' + strftime(" %H:%M:%S", localtime()) + '\n')
        print("TRIP!\n")

    else:
        tripBOOL.write('0')

    tripBOOL.close()
        

    time.sleep(0.1)
