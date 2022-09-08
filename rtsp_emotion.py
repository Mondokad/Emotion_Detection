from cProfile import label
from cgitb import text
from distutils import text_file
import time
from tkinter import Label

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

import sys
import configparser
import time
import numpy as np
import imutils
import cv2
from pydub import AudioSegment
from pydub.playback import play
import paho.mqtt.client as mqttClient

import sounddevice as sd
from numpy import linalg as LA

from nanoleafapi import Nanoleaf
import requests
from nanoleafapi import RED, ORANGE, YELLOW, GREEN, LIGHT_BLUE, BLUE, PINK, PURPLE, WHITE

nl = Nanoleaf("192.168.128.10", "NRphdmYhP9Gnqy5LtqQztRWebsnjge8D")
nl.set_color(WHITE)
nl.power_off()
import winsound
duration_sound = 1000  # milliseconds
freq = 440  # Hz


### Gather configuration parameters
def gather_arg():

    conf_par = configparser.ConfigParser()
    try:
        conf_par.read('credentials.ini')
        host= conf_par.get('camera', 'host')
        broker = conf_par.get('mqtt', 'broker')
        port = conf_par.getint('mqtt', 'port')
        prototxt = conf_par.get('ssd', 'prototxt')
        model = conf_par.get('ssd', 'model')
        conf = conf_par.getfloat('ssd', 'conf')
    except:
        print('Missing credentials or input file!')
        sys.exit(2)
    return host, broker, port, prototxt, model, conf

## connect to MQTT Broker ###
#def on_connect(client, userdata, flags, rc):
#    if rc == 0:
#        print("Connected to broker")
#        global Connected                #Use global variable
#        Connected = True                #Signal connection
#    else:
#        print("Connection failed")

#(host, broker, port, prototxt, model, conf) = gather_arg()

#Connected = False   #global variable for the state of the connection
#client = mqttClient.Client("Python")               #create new instance
#client.on_connect= on_connect                      #attach function to callback
#client.connect(broker, port=port)          #connect to broker
#client.loop_start()        #start the loop
#while Connected != True:    #Wait for connection
#    time.sleep(1.0)

 # Load the model
model = Sequential()
classifier = load_model('ferjj.h5') # This model has a set of 6 classes

# We have 6 labels for the model
class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
classes = list(class_labels.values())
# print(class_labels)


face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')




# This function is for designing the overlay text on the predicted image boxes.
def text_on_detected_boxes(text,text_x,text_y,image,font_scale = 1,
                           font = cv2.FONT_HERSHEY_SIMPLEX,
                           FONT_COLOR = (0, 0, 0),
                           FONT_THICKNESS = 2,
                           rectangle_bgr = (0, 255, 0)):



    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    # Set the Coordinates of the boxes
    box_coords = ((text_x-10, text_y+4), (text_x + text_width+10, text_y - text_height-5))
    # Draw the detected boxes and labels
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y), font, fontScale=font_scale, color=FONT_COLOR,thickness=FONT_THICKNESS)


# Detection of the emotions on an image:

def face_detector_image(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) # Convert the image into GrayScale image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x, w, y, h))
    return rects, allfaces, img


def emotionImage(imgPath):
    img = cv2.imread(imgPath)
    rects, faces, image = face_detector_image(img)

    i = 0
    for face in faces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        print(preds)
        label = class_labels[preds.argmax()]
        label_position = (rects[i][0] + int((rects[i][1] / 2)), abs(rects[i][2] - 10))
        i = + 1

        # Overlay our detected emotion on the picture

        text_on_detected_boxes(label, label_position[0],label_position[1], image)


    cv2.imshow("Emotion Detector", image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()



# Detection of the expression on video stream
def face_detector_video(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y + h, x:x + w]

    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    return (x, w, y, h), roi_gray, img





def emotionVideo(cap):
    angyry_counter = []
    while True:

        ret, frame = cap.read()
        rect, face, image = face_detector_video(frame)
        if np.sum([face]) != 0.0:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (rect[0] + rect[1]//50, rect[2] + rect[3]//50)

            text_on_detected_boxes(label, label_position[0], label_position[1], image) # You can use this function for your another opencv projects.
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(image, str(fps),(5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if label == "Angry":
                angyry_counter.append("Angry")
                print(angyry_counter)
                if len(angyry_counter)==100: #approximately 30 seconds being angry for the duration of the program.
                    print("you are angry")                   
                    winsound.Beep(freq, duration_sound)
                    angyry_counter.clear()
                    nl.set_color(RED)
                    time.sleep(3)
                    nl.set_color(WHITE)
                
                    
                
  
        else:
            cv2.putText(image, "No Face Found", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('All', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    camera = cv2.VideoCapture(0) # If you are using a USB Camera then use 1 instead of 0. or 'rtsp://192.168.128.4:9000/live' for a live camera feed without MQTT
    emotionVideo(camera)
        
    #IMAGE_PATH = "provide the image path"
    # emotionImage(IMAGE_PATH) # If you are using this on an image please provide the path
