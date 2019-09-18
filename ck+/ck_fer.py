#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:44:31 2019

@author: austin
"""
import os
import re
import numpy as np
import cv2
import pre_processing
import model_gen
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
cohn-kanade-images/S005/001/S005_001_00000011.png
Emotion/S005/001/S005_001_00000011_emotion.txt
'''
label_names = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']
BASE_DIR = DATA_DIR + '/ck/CK+/'
EMO_DIR = BASE_DIR + 'Emotion_labels/Emotion/'
IMG_DIR = BASE_DIR + 'extended-cohn-kanade-images/cohn-kanade-images/'
CASCADE_DIR = PROJECT_DIR + '/FaceDetection/xml'
FACE_XML = CASCADE_DIR + '/haarcascade_frontalface_default.xml'
EYE_XML = CASCADE_DIR + '/haarcascade_eye.xml'

FACE_CASCADE = cv2.CascadeClassifier(FACE_XML)
EYE_CASCADE = cv2.CascadeClassifier(EYE_XML)

def load_data():
    data = {}
    for subject_dir in os.listdir(EMO_DIR):
        for inst_dir in os.listdir(EMO_DIR + subject_dir + '/'):
            emotion_file_names = os.listdir(EMO_DIR + subject_dir + '/' + inst_dir + '/')
            if not emotion_file_names:
                continue
            emotion_file_name = emotion_file_names[0]

            img_file_name = re.sub('_emotion.txt', '.png', emotion_file_name)
            image_file_path = IMG_DIR + subject_dir + '/' + inst_dir + '/' +  img_file_name
            emotion_file_path = EMO_DIR + subject_dir + '/' + inst_dir + '/' +  emotion_file_name

            emotion_score = ''
            with open(emotion_file_path, 'r') as f:
                emotion_score = eval(f.readline())

            data[image_file_path] = emotion_score
    return data

def load_images(img_height, img_width, img_chan, edge_images=False):
    data = load_data()

    num_images = len(data)

    images = np.empty((num_images, img_height, img_width), dtype=np.uint8)
    labels = np.empty((num_images), dtype=np.uint8)
    file_count = 0
    for file_name, label in data.items():
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        faces = FACE_CASCADE.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            face_img = img[y:y+h, x:x+w]
        img = cv2.resize(face_img, (img_height, img_width))
        if edge_images:
            img = pre_processing.edge_detection(img)
        images[file_count] = img
        labels[file_count] = int(label)
        file_count += 1


    return images, labels

if __name__ == '__main__':
    img_height = 128
    img_width = 128
    images, labels = load_images(img_height, img_width, 1, edge_images=False)

    n_classes = len(np.unique(labels))
    print(f'There are {n_classes} classes.')

    print(images.shape)
    print(labels.shape)
    print(np.unique(labels))

    model = model_gen.create_wide_model(img_height, img_width, 1, 20, n_classes)
    #model = model_gen.create_deep_model(128, 128, 1, 20, len(np.unique(labels)))

    X_test = images
    X_train = images
    y_train = labels
    y_test = labels


    history = model.fit([images, images], labels,
                    batch_size=10,
                    epochs=70
                    ,validation_split=0.2)

    test_scores = model.evaluate([images, images], y_test, verbose=0)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()