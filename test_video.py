#usage 
#python test_video.py --model numeral.model
#python test_video.py --model numeral_color.model

#import libaries
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

width = 640
height = 480
threshold = 0.75
cameraNo = 1

capture = cv2.VideoCapture(cameraNo)
capture.set(3, width)
capture.set(4, height)

model = load_model(args["model"])

def preProcessing(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img/255

        return img

while True:
    success, imgOrg = capture.read()
    img = np.asarray(imgOrg)
    img = cv2.resize(img, (32,32))
    img = preProcessing(img)

    #cv2.imshow("Processed Img", img)
    img = img.reshape(1,32,32,1)

    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    #print(classIndex)
    #print(predictions)

    probVal =  np.amax(predictions)
    print(classIndex,probVal)

    if probVal > threshold:
            cv2.putText(imgOrg, str(classIndex) + "  " + str(probVal),
                       (50,50), cv2.FONT_HERSHEY_COMPLEX,
                       1, (0,255,0), 2)
    cv2.imshow("Original Img", imgOrg)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

print("[INFO] cleaning up...")
cv2.destroyAllWindows()
         
