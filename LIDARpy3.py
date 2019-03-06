# -*- coding: utf-8 -*
import serial
import time
from time import strftime, localtime



initTime=int(time.time())



ser = serial.Serial('/dev/serial0', 115200, timeout = 1)

def getTFminiData():
    #while True:
    tripValue = 800
    tripCounter = 0
    while int(time.time())-initTime<=900:
        #time.sleep(0.1)
        count = ser.in_waiting
        if count > 8:
            recv = ser.read(9)   
            ser.reset_input_buffer() 
            # type(recv), 'str' in python2(recv[0] = 'Y'), 'bytes' in python3(recv[0] = 89)
            # type(recv[0]), 'str' in python2, 'int' in python3 
            
            if recv[0] == 0x59 and recv[1] == 0x59:     #python3
                f=open('/home/pi/Desktop/FINALBRANCH/lidardata.txt','a')
                f2=open('/home/pi/Desktop/FINALBRANCH/lidarDISTANCEREADING.txt','a')
                tripBOOL = open('/home/pi/Desktop/FINALBRANCH/isTripped.txt', 'w')
                tripCountFile = open('/home/pi/Desktop/FINALBRANCH/timesTripped.txt', 'a')
                distance = recv[2] + recv[3] * 256
                strength = recv[4] + recv[5] * 256
                temp = str(distance)+'\t'+str(strength)
                distanceTRIPTEST = str(distance)
                temp = temp+'\t'+strftime(" %H:%M:%S", localtime())
                #####print(temp)
                #f.write('test\n')

                if (distance < tripValue):
                    tripBOOL.write('1')
                    tripCounter += 1
                    #tripString = tripCounter + '\n'
                    tripCountFile.write(str(tripCounter) + '\t@\t' + strftime(" %H:%M:%S", localtime()) + '\n')
                else:
                    tripBOOL.write('0')
                    

             
                
                f.write(temp+'\n')
                f2.write(distanceTRIPTEST+'\n')
                time.sleep(0.1)
                f.close()
                f2.close()
                ser.reset_input_buffer()
                
            #if recv[0] == 'Y' and recv[1] == 'Y':     #python2
             #   lowD = int(recv[2].encode('hex'), 16)      
              #  highD = int(recv[3].encode('hex'), 16)
               # lowS = int(recv[4].encode('hex'), 16)      
                #highS = int(recv[5].encode('hex'), 16)
                #distance = lowD + highD * 256
                #strength = lowS + highS * 256
                #print(distance, strength)
            
            # you can also distinguish python2 and python3: 
            #import sys
            #sys.version[0] == '2'    #True, python2
            #sys.version[0] == '3'    #True, python3


if __name__ == '__main__':
    try:
        if ser.is_open == False:
            ser.open()
        #path = '/home/pi/Desktop/LIDAR/currentBranch/lidardata.txt'   
        getTFminiData()
    except KeyboardInterrupt:   # Ctrl+C
        if ser != None:
            ser.close()
