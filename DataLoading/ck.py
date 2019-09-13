#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:58:09 2019

@author: austin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:44:31 2019

@author: austin
"""

import os
import cv2
import sys
#import tensorflow as tf
import numpy as np
import scipy.fftpack
import re
import model_gen
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


print("OpenCV Version:", cv2.__version__)





'''
0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
cohn-kanade-images/S005/001/S005_001_00000011.png
Emotion/S005/001/S005_001_00000011_emotion.txt
'''
DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']
base_dir = DATA_DIR + '/ck/CK+/'
emo_dir = base_dir + 'Emotion_labels/Emotion/'
img_dir = base_dir + 'extended-cohn-kanade-images/cohn-kanade-images/'
cascade_dir = PROJECT_DIR + '/FaceDetection/xml'
face_cascade_xml = cascade_dir + '/haarcascade_frontalface_default.xml'
eye_cascade_xml = cascade_dir + '/haarcascade_eye.xml'

def load_data():
    data = {}
    for subject_dir in os.listdir(emo_dir):
        for inst_dir in os.listdir(emo_dir + subject_dir + '/'):
            emotion_file_names = os.listdir(emo_dir + subject_dir + '/' + inst_dir + '/')
            if not emotion_file_names:
                continue
            emotion_file_name = emotion_file_names[0]
            #print('Emotion File:', emotion_file_name)
            img_file_name = re.sub('_emotion.txt', '.png', emotion_file_name)
            image_file_path = img_dir + subject_dir + '/' + inst_dir + '/' +  img_file_name
            emotion_file_path = emo_dir + subject_dir + '/' + inst_dir + '/' +  emotion_file_name
            #print(os.path.isfile(image_file_path))
            emotion_score = ''
            with open(emotion_file_path, 'r') as f:
                emotion_score = eval(f.readline())
            #print('Image:', img_file_name, ', emotion:', emotion_score)
            data[image_file_path] = emotion_score
    return data

def visualize(data):
    #print(data)
    words = np.array(list(data.values()))
    print(Counter(words).keys()) # equals to list(set(words))
    print(Counter(words).values()) # counts the elements' frequency

    plt.hist(words, bins=np.arange(words.min(), words.max()+1))




#label_names = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

face_cascade = cv2.CascadeClassifier(face_cascade_xml)
eye_cascade = cv2.CascadeClassifier(eye_cascade_xml)

def top_eyes(eyes):
    top_eyes = []
    for eye in eyes:
        if len(top_eyes) == 0:
            top_eyes.append(eye)
            continue
        if len(top_eyes) == 1:
            top_eyes.append(eye)
            if eye[0] < top_eyes[0][0]:
                top_eyes[1] = top_eyes[0]
                top_eyes[0] = eye
            continue
        d1 = abs(eye[1] - top_eyes[0][1])
        d2 = abs(eye[1] - top_eyes[1][1])
        d3 = abs(top_eyes[0][1] - top_eyes[1][1])
        min_dist = min(d1, d2, d3)
        if d1 == min_dist:
            if top_eyes[0][0] > eye[0]:
                top_eyes[1] = top_eyes[0]
                top_eyes[0] = eye
            else:
                top_eyes[1] = eye
        elif d2 == min_dist:
            if top_eyes[1][0] < eye[0]:
                top_eyes[0] = top_eyes[1]
                top_eyes[1] = eye
            else:
                top_eyes[0] = eye
        elif d3 == min_dist:
            continue
    return top_eyes

def create_dataset(data):
    #img_dir = '/home/austin/Desktop/gPrj/data/jaffe/'
    try_eyes = False
    show_img = False
    write_image = True

    num_images = len(data) - 1

    images = []
    labels = []
    images = np.zeros((num_images,128,128,1))
    freq_images = np.zeros((num_images,128,128,1))
    file_count = 0
    for file_name, label in data.items():
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:

            face_img = img[y:y+h, x:x+w]
            face_img = cv2.GaussianBlur(face_img,(3,3),0)
            face_img = cv2.Laplacian(face_img, cv2.CV_64F)

            face_img = cv2.resize(face_img, (128, 128))

            if write_image:
                #freq_img = np.fft.fft2(img)
                freq_img = scipy.fftpack.dct(face_img)
                scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
                scaler.fit(freq_img)
                scaler.transform(freq_img)
                face_img = np.expand_dims(face_img, axis=2)
                freq_img = np.expand_dims(freq_img, axis=2)
                images[file_count]=face_img
                freq_images[file_count]=freq_img
                labels.append(int(label) - 1)

                if try_eyes:
                    eyes = eye_cascade.detectMultiScale(face_img)

                    # If more than 2 eyes find the two closest on the y plane
                    if len(eyes) > 2:
                        #eyes = top_eyes(eyes)
                        indxs = []
                        for eye in eyes:
                            indxs.append(eye[1])
                        print(type(indxs))
                        print(indxs)
                        val, idx1 = min((val, idx) for (idx, val) in enumerate(indxs))
                        print(val, idx1)
                        indxs[idx1] = max(indxs)
                        val, idx2 = min((val, idx) for (idx, val) in enumerate(indxs))
                        print(val, idx2)
                        tmp = [eyes[idx1], eyes[idx2]]
                        if tmp[0][0] > tmp[1][0]:
                            tmp2 = tmp
                            tmp = [tmp2[1], tmp2[0]]
                        eyes = tmp
                    if len(eyes) == 2:
                        if eyes[0][0] > eyes[1][0]:
                                tmp2 = eyes
                                eyes = [tmp2[1], tmp2[0]]

                    count = 1
                    for (ex,ey,ew,eh) in eyes:
                        eye_img = face_img[ey:ey+eh, ex:ex+ew]
                        #print(eye_img.shape)
                        cv2.rectangle(face_img,(ex,ey),(ex+ew,ey+eh),(count * 255,255,0),2)
                        cv2.imshow("cropped", eye_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        count -= 1
            if show_img:
                cv2.imshow("cropped", face_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            file_count += 1
            #if file_count > 20:

    labels = np.array(labels)
    return images, labels, freq_images


data = load_data()
visualize(data)
images, labels, freq_images = create_dataset(data)
print(images.shape)
print(labels.shape)
print(freq_images.shape)
print(np.unique(labels))

if True:
    model = model_gen.create_deep_model(128, 128, 1, 20, len(np.unique(labels)))
    x = images.reshape(len(images), 128, 128, 1).astype('float32') / 255
    xfq = freq_images.reshape(len(images), 128, 128, 1).astype('float32')

    y = labels

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=False)
    Xf_train, Xf_test, y_train, y_test = train_test_split(xfq, y, train_size=0.75, shuffle=False)

    #x_test_sp = images.reshape(len(images), 128, 128, 1).astype('float32') / 255
    #x_test_fq = freq_images.reshape(len(images), 128, 128, 1).astype('float32')
    #y_test = labels

    history = model.fit([X_train, X_train], y_train,
                        batch_size=20,
                        epochs=200)
                        #,validation_split=0.2)

    test_scores = model.evaluate([X_test, X_test], y_test, verbose=0)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
