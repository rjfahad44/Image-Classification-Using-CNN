import random as rn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, concatenate, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
import os
import pathlib
import cv2
from tqdm import tqdm
import os
from zipfile import ZipFile
from PIL import Image
from os import listdir
from os.path import join
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import preprocessing


np.random.seed(42)

# hyper-parameters
batch_size = 64

# categories of images
num_classes = 2

# number of training epochs
epochs = 200

IMG_SIZE = 224


def load_data2():
    """This function loads dataset, normalized, and labels one-hot encoded"""
    train_data_dir = pathlib.Path('MyDataSet/Train')
    test_data_dir = pathlib.Path('MyDataSet/Test')
    train_folders = os.listdir(train_data_dir)
    test_folders = os.listdir(test_data_dir)

    train_image_names = []
    test_image_names = []
    train_labels = []
    train_images = []
    test_labels = []
    test_images = []

    size = 224, 224

    for folder in train_folders:
        for file in os.listdir(os.path.join(train_data_dir,folder)):
            if file.endswith("jpg"):
                train_image_names.append(file)
                train_labels.append(folder)
                img = cv2.imread(os.path.join(train_data_dir,folder,file))
                im = cv2.resize(img,size)
                train_images.append(im)
            else:
                continue

    for folder in test_folders:
        for file in os.listdir(os.path.join(test_data_dir,folder)):
            if file.endswith("jpg"):
                test_image_names.append(file)
                test_labels.append(folder)
                img = cv2.imread(os.path.join(test_data_dir,folder,file))
                im = cv2.resize(img,size)
                test_images.append(im)
            else:
                continue

    X_train = np.array(train_images)
    X_test = np.array(test_images)
 
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    train_label_dummies = pandas.get_dummies(train_labels)
    test_label_dummies = pandas.get_dummies(test_labels)

    trainlabels = train_label_dummies.values.argmax(1)
    testlabels = test_label_dummies.values.argmax(1)

    X_train = np.array(X_train)
    y_train = np.array(trainlabels)


    X_test = np.array(X_test)
    y_test = np.array(testlabels)

    print('OK')
    return(X_train, y_train),(X_test, y_test), (train_image_names, test_image_names)

def create_model():
  model = Sequential()

  model.add(Conv2D(224, (3, 3), activation='relu', input_shape=(224, 224, 3)))
  model.add(MaxPooling2D(pool_size=(3, 3)))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3, 3)))

  #model.add(Dropout(0.25))    
  model.add(Flatten())

  model.add(Dense(num_classes, activation='softmax'))
  model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

  return model
	
if __name__ == "__main__":

    # load the data
    (X_train, y_train), (X_test, y_test), (train_image_names, test_image_names) = load_data2()
	
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=1)
    #print("Validation samples:", X_val.shape[0])
    
    # construct the model
    model = create_model()
    
    # train
    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=2,
            validation_steps=2,
            validation_data=(X_val, y_val))
    validation_steps = 20
    
    # evaluate
    loss0, accuracy0 = model.evaluate(X_val, y_val, steps = 20)

    print("Validation Loss: {:.2f}".format(loss0))
    print("Validation Accuracy: {:.2f}".format(accuracy0))

# test item prediction
testLabelPredicted = model.predict(X_test)
testLabelPredicted =  np.rint(testLabelPredicted.argmax(axis=-1))
#print(test_image_names)
testLabelGold = y_test
#print(testLabelGold)

# Evaluation
results = confusion_matrix(testLabelGold, testLabelPredicted) 
    
print ('Confusion Matrix :')
print (results) 

print ('Recall Score :',recall_score(testLabelGold, testLabelPredicted, average='micro'))
print ('Precision Score :',precision_score(testLabelGold, testLabelPredicted, average='micro'))
print ('F1 Score :',f1_score(testLabelGold, testLabelPredicted, average='micro'))
print ('Accuracy :',accuracy_score(testLabelGold, testLabelPredicted))

