
from __future__ import division, print_function
import time
from tfmini import TFmini
print('Hello')

tf = TFmini('/dev/ttyAMA0', mode=TFmini.STD_MODE)

try:
    
    print('='*25)
    while True:
        d = tf.read()
        if d:
            print('Distance: {:5}'.format(d))
        else:
            print('No valid response')
        time.sleep(0.1)

except KeyboardInterrupt:
    tf.close()
    print('bye!!')


