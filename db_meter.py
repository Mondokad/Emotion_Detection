import sounddevice as sd
from numpy import linalg as LA
import numpy as np

duration = 10000  # milliseconds

def print_sound(indata, outdata, frames, time, status):

    volume_norm = np.linalg.norm(indata)*10
    print(volume_norm)
    arr = int(volume_norm)
    # print (arr)
    if volume_norm > 100:
        print('LOUD')
    # return volume_norm


with sd.Stream(callback=print_sound):
    sd.sleep(duration)

   
