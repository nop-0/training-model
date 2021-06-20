#python training.py --path DATASET --model test.model

import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout,Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, default="",
	help="path to dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
args = vars(ap.parse_args())

##########################
#path = "DATASET"
###########################
images = []
classNo = []
testRatio = 0.2
validRatio = 0.2
imgDimensions = (32,32,3)
batchSizeVal = 32 #number of samples in an epoch used to estimate model error
epochsVal = 100 #no of loop through the training datase
stepsPerEpoch = 200 #based on how much data the algorithm can extract


myList = os.listdir(args["path"])
no_classes = len(myList)
print("Number of classes detected:", no_classes)

######Importing classes##############
for x in range (0,no_classes):
    myDatalist = os.listdir(args["path"] + "/" + str(x)) #path to img data folder
    for y in myDatalist:
        curImg = cv.imread(args["path"] + "/" + str(x) + "/" + str(y))#path to each img
        curImg = cv.resize(curImg, (imgDimensions[0],imgDimensions[1]))
        images.append(curImg) #stores image data in image list
        classNo.append(x)
    print(x, end = " ")
print(" ")

images = np.array(images)
classNo = np.array(classNo)

#print(images.shape)
print("Number of classNo :",classNo.shape)

#####Data spliting#######
#X for image, Y for image ID
X_train,X_test,Y_train,Y_test = train_test_split(images,classNo,test_size = testRatio)
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size = validRatio)

#print(X_train.shape)
#print(X_test.shape)
#print(X_validation.shape)

numOfSamples = []

for x in range(0,no_classes):
    #print(len(np.where(Y_train==0)[0]))
    numOfSamples.append(len(np.where(Y_train==x)[0]))

print(numOfSamples)

plt.figure(figsize = (10,5))
plt.bar(range(0,no_classes), numOfSamples)
plt.title("No of Img for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Img")
plt.show()

def preProcessing(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img

#img = X_train[30]
#img = cv.resize(img, (300,300))
#cv.imshow("preprocessed",img)
#cv.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_test.shape[2],1) #add depth of 1 into X_train.shape
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)


dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.2,
                             shear_range = 0.1,
                             rotation_range = 10)

dataGen.fit(X_train)
Y_train = to_categorical(Y_train, no_classes)
Y_test = to_categorical(Y_test, no_classes)
Y_validation = to_categorical(Y_validation, no_classes)

def myModel():
    noOfFilters = 60 #based on LENET model
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imgDimensions[0],
                                                             imgDimensions[1],
                                                             1),activation = 'relu'
                                                             )))
    
    model.add((Conv2D(noOfFilters,sizeOfFilter1,activation = 'relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation = 'relu')))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation = 'relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes,activation = 'softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = myModel()
print(model.summary())

history = model.fit(dataGen.flow(X_train,Y_train,
                                 batch_size = batchSizeVal),
                                 steps_per_epoch = stepsPerEpoch,
                                 epochs = epochsVal,
                                 validation_data = (X_validation,Y_validation),shuffle=1)
    
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test,Y_test, verbose = 0)
print('Test score = ' , score[0])
print('Test accuracy =', score [1])

model.save(args["model"], save_format="h5")
