import serial
import time
import RPi.GPIO as GPIO

LEDpin = 11

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(LEDpin,GPIO.OUT)
GPIO.output(LEDpin,GPIO.LOW)

ser = serial.Serial('/dev/serial0',115200,timeout = 1)


ser.write(0x42) #1
#ser.write(bytes(b'B'))

ser.write(0x57) #2
#ser.write(bytes(b'W'))

ser.write(0x02) #3
#ser.write(bytes(2))

ser.write(0x00) #4
#ser.write(bytes(0))
 
ser.write(0x00) #5
#ser.write(bytes(0))

ser.write(0x00) #6
#ser.write(bytes(0))
          
ser.write(0x01) #7
#ser.write(bytes(1))
          
ser.write(0x06) #8
#ser.write(bytes(6))

#BAUD SET
ser.write(0x42) #1
ser.write(0x57) #2
ser.write(0x02) #3
ser.write(0x00) #4
ser.write(0x00) #5
ser.write(0x00) #6
ser.write(0x06) #7
ser.write(0x08) #8

#BAUD SET
ser.write(0x42) #1
ser.write(0x57) #2
ser.write(0x02) #3
ser.write(0x00) #4
ser.write(0x00) #5
ser.write(0x00) #6
ser.write(0x06) #7 SETTING: 115200
ser.write(0x08) #8

#DISTANCE SET
#SHORT: 0x02
#LONG: 0x07
ser.write(0x42) #1
ser.write(0x57) #2
ser.write(0x02) #3
ser.write(0x00) #4
ser.write(0x00) #5
ser.write(0x00) #6
ser.write(0x02) #7 SETTING: LONG MODE
ser.write(0x11) #8


while(True):
    
    while(ser.in_waiting >= 9):
        #print (ser.read())
        if((b'Y' == ser.read()) and ( b'Y' == ser.read())):
            
            GPIO.output(LEDpin, GPIO.LOW)
            Dist_L = ser.read()
            Dist_H = ser.read()
            Dist_Total = (ord(Dist_H) * 256) + (ord(Dist_L))
            for i in range (0,5):
                ser.read()
                
            print(Dist_Total)
            #if(Dist_Total < 20):
                #GPIO.output(LEDpin, GPIO.HIGH)


        
