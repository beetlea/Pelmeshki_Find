import numpy as np
import cv2

from keras.models import load_model
import time

def print_class(data):
    data_ = data
    data_out1 = data[0,0]
    data_out = 0
    print(data)
    for i in range(4):
        if data_out1 < data_[0,i]:
            data_out = i
            data_out1 = data[0,i]
    print(data_[0,data_out])
    if data_out == 0:
        return 'borsh'
    if data_out == 1:
        return 'tea'
    if data_out == 2:
        return 'coffe'
    if data_out == 3:
        return 'pelmen'

image = np.zeros((800,1000,3), np.uint8)

	
# Window name in which image is displayed 
window_name = 'Image'

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 

# org 
otstup = 150
org = (50, otstup) 
org1 = (50, 400) 
# fontScale 
fontScale = 5

# Blue color in BGR 
color = (255, 255, 255) 
balans =2000
# Line thickness of 2 px 
thickness = 3
cv2.imshow("food", image) 
model2= load_model('model_vgg19_15epochs.h5')
cap = cv2.VideoCapture(1)

while(True):
# Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape)
    if ret == True:
        #cv2.imshow('first tarelka',frame)
        if cv2.waitKey(33) == ord('d'):

            tarelka = cv2.resize(frame,(224,224))
            #cv2.imshow('first tarelka',frame)

            mmm = model2.predict(np.reshape(tarelka,(1,224,224,3)))
            name = print_class(mmm)
            image = cv2.putText(image, name, org, font, 
				fontScale, color, thickness, cv2.LINE_AA) 
            otstup = otstup + 150
            org = (50, otstup)
            cv2.imshow("food", image) 
            time.sleep(1)
        cv2.imshow("video", frame) 
        """
        tarelka = cv2.resize(frame[0:300,0:340,:],(224,224))
        print("тарелка = ")
        mmm = model.predict(np.reshape(tarelka,(1,224,224,3)))
        print_class(mmm)
        print("кружка = ")
        #cv2.imshow('two_tarelka', frame[240:480, 340: 640,:])
        #tarelka2 = cv2.resize(frame[0:240,0:340,:],(224,224))
        #ynew = model.predict_classes(tarelka2)
        cv2.imshow('kryzhka', frame[120:480, 300:600,:])
        kryzhka = cv2.resize(frame[120:480, 300:600,:],(224,224))
        hhhh = model.predict(np.reshape(kryzhka,(1,224,224,3)))
        print_class(hhhh)
        """
    if cv2.waitKey(33) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
