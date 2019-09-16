#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:44:31 2019

@author: austin
"""

import os
import cv2
import tensorflow as tf
import numpy as np
import scipy.fftpack
import re
import model_gen
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print("OpenCV Version:", cv2.__version__)


label_names = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']

cascade_dir = PROJECT_DIR + '/FaceDetection/xml'
face_cascade_xml = cascade_dir + '/haarcascade_frontalface_default.xml'
eye_cascade_xml = cascade_dir + '/haarcascade_eye.xml'


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

def for_jaffe(h, w, c):
    ''' height, width, channels'''
    img_dir = DATA_DIR + '/jaffe/'
    tiff_pattern = re.compile('\.tiff', re.IGNORECASE)
    hap_ptr = re.compile('HA')
    sad_ptr = re.compile('SA')
    sur_ptr = re.compile('SU')
    ang_ptr = re.compile('AN')
    dis_ptr = re.compile('DI')
    fea_ptr = re.compile('FE')
    neu_ptr = re.compile('NE')
    patterns = [hap_ptr, sad_ptr, sur_ptr, ang_ptr, dis_ptr, fea_ptr, neu_ptr]
    try_eyes = False
    show_img = False
    write_image = True


    images = []
    labels = []
    images = np.zeros((213, h, w, c)) # 213 images in dataset
    freq_images = np.zeros((213, h, w, c))
    file_count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            img = cv2.imread(img_dir + file_name, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                face_img = img[fy:fy+fh, fx:fx+fw]
                face_img = cv2.resize(face_img, (h, w))
                if write_image:
                    for idx, pattern in enumerate(patterns):
                        m = pattern.search(file_name)
                        if m:
                            face_img = np.expand_dims(face_img, axis=2)
                            #freq_img = np.fft.fft2(img)
                            freq_img = scipy.fftpack.dct(face_img)
                            images[file_count] = face_img
                            freq_images[file_count] = freq_img
                            labels.append(idx)
                            break

                if try_eyes:
                    eyes = eye_cascade.detectMultiScale(face_img)
                    # If more than 2 eyes find the two closest on the y plane
                    if len(eyes) > 2:
                        #eyes = top_eyes(eyes)
                        indxs = []
                        for eye in eyes:
                            indxs.append(eye[1])
                        val, idx1 = min((val, idx) for (idx, val) in enumerate(indxs))
                        indxs[idx1] = max(indxs)
                        val, idx2 = min((val, idx) for (idx, val) in enumerate(indxs))
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
                    for (ex, ey, ew, eh) in eyes:
                        eye_img = face_img[ey:ey+eh, ex:ex+ew]
                        #print(eye_img.shape)
                        cv2.rectangle(face_img, (ex, ey), (ex+ew, ey+eh), (count * 255, 255, 0), 2)
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

if __name__ == '__main__':
    HEIGHT, WIDTH, CHANNELS = 128, 128, 1
    images, labels, freq_images = for_jaffe(HEIGHT, WIDTH, CHANNELS)


    model = model_gen.create_wide_model(HEIGHT, WIDTH, CHANNELS, 50, len(np.unique(labels)))
    x_train_sp = images.reshape(len(images), HEIGHT, WIDTH, CHANNELS).astype('float32') / 255
    x_train_fq = freq_images.reshape(len(images), HEIGHT, WIDTH, CHANNELS)
    y_train = labels

    x_test_sp = images.reshape(len(images), HEIGHT, WIDTH, CHANNELS).astype('float32') / 255
    x_test_fq = freq_images.reshape(len(images), HEIGHT, WIDTH, CHANNELS)
    y_test = labels

    history = model.fit([x_train_sp, x_train_sp], y_train,
                        batch_size=10,
                        epochs=200,
                        validation_split=0.2)

    test_scores = model.evaluate([x_test_sp, x_test_sp], y_test, verbose=0)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    print(np.unique(labels))
