from gc import callbacks
import sounddevice as sd
from numpy import linalg as LA
import numpy as np

duration = 20  # seconds

def print_sound(indata, outdata, frames, time, status):

    volume_norm = np.linalg.norm(indata)*10
    global arr
    arr = int(volume_norm)
    print (arr)
   
with sd.Stream(callback=print_sound):
    sd.sleep(duration * 1000)

#newlist = []
#newlist.append(arr)
#if len(newlist) == 10:
#    print(newlist)
