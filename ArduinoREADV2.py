# -*- coding: utf-8 -*
import serial
import time
from time import strftime, localtime

ser = serial.Serial('/dev/ttyACM0', 115200, timeout = 1) # ttyACM0 for Arduino board

readOut = 0   #chars waiting from laser range finder

print ("Starting up")
connected = False
initTime=int(time.time())
#commandToSend = 1 # get the distance in mm

#while True:
while int(time.time())-initTime<=900:
    #print ("Writing: ",  commandToSend)
    #ser.write(str(commandToSend).encode())
    #time.sleep(1)
    while True:
        try:
            #print ("Attempt to Read")
            readOut = strftime(' %H:%M:%S', localtime()) + '  ' + ser.readline().decode('ascii')
            time.sleep(0.1)
            #print ("Reading: ", readOut)
            
            #print('wow')
            #####print (readOut)
            
                
                   
            break
        except:
            pass
    f=open('/home/pi/Desktop/FINALBRANCH/PIR.txt','a')
    f.write(readOut)
    f.close()
    #print ('\t'+strftime(' %H:%M:%S', localtime()))
    #print ('\n')
    #print ("Restart")
    ser.flush() #flush the buffer
