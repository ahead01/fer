#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:42:31 2019

@author: austin
"""

import os
import cv2
import tensorflow as tf
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
ck: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
JAFEE: 

'''
ck_label_names = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

combined_label_names = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']

cascade_dir = PROJECT_DIR + '/FaceDetection/xml'
face_cascade_xml = cascade_dir + '/haarcascade_frontalface_default.xml'
eye_cascade_xml = cascade_dir + '/haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_xml)
eye_cascade = cv2.CascadeClassifier(eye_cascade_xml)


def load_jaffe(image_height, image_width, channels):
    img_dir = DATA_DIR + '/jaffe/'
    tiff_pattern = re.compile('\.tiff', re.IGNORECASE)
    jaffe_img_count = 213
    hap_ptr = re.compile('HA')
    sad_ptr = re.compile('SA')
    sur_ptr = re.compile('SU')
    ang_ptr = re.compile('AN')
    dis_ptr = re.compile('DI')
    fea_ptr = re.compile('FE')
    neu_ptr = re.compile('NE')
    patterns= [neu_ptr, ang_ptr, dis_ptr, fea_ptr, hap_ptr, sad_ptr, sur_ptr]
    patterns = [hap_ptr, sad_ptr, sur_ptr, ang_ptr, dis_ptr, fea_ptr, neu_ptr]

    images = []
    labels = []
    images = np.zeros((jaffe_img_count, image_height, image_width, channels))
    freq_images = np.zeros((jaffe_img_count, image_height, image_width, channels))

    file_count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            img = cv2.imread(img_dir + file_name, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (image_height, image_width))
                for idx, pattern in enumerate(patterns):
                    m = pattern.search(file_name)
                    if m:
                        img = np.expand_dims(face_img, axis=2)
                        #freq_img = np.fft.fft2(img)
                        freq_img = scipy.fftpack.dct(img)
                        images[file_count]=img
                        freq_images[file_count]=freq_img
                        labels.append(idx)
                        file_count += 1
                        break
    labels = np.array(labels)
    return [images, labels, freq_images]

def load_ck_paths():
    base_dir = DATA_DIR + '/ck/CK+/'
    emo_dir = base_dir + 'Emotion_labels/Emotion/'
    img_dir = base_dir + 'extended-cohn-kanade-images/cohn-kanade-images/'
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

def load_ck(image_height, image_width, channels):
    data = load_ck_paths()
    num_images = len(data) - 1

    images = []
    labels = []
    images = np.zeros((num_images, image_height, image_width, channels))
    freq_images = np.zeros((num_images, image_height, image_width, channels))
    file_count = 0
    for file_name, label in data.items():
        if int(label) == 2:
            continue
        if int(label) > 2:
            label = int(label) - 1

        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.GaussianBlur(face_img, (3,3),0)
            face_img = cv2.Laplacian(face_img, cv2.CV_64F)
            face_img = cv2.resize(face_img, (image_height, image_width))

            freq_img = scipy.fftpack.dct(face_img)
            scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
            scaler.fit(freq_img)
            scaler.transform(freq_img)
            face_img = np.expand_dims(face_img, axis=2)
            freq_img = np.expand_dims(freq_img, axis=2)
            images[file_count]=face_img
            freq_images[file_count]=freq_img
            labels.append(int(label) - 1)

    labels = np.array(labels)
    return [images, labels, freq_images]


def load_combined(image_height, image_width, channels):
    ck_data = load_ck(image_height, image_width, channels)
    jafee_data = load_jaffe(image_height, image_width, channels)
    print(len(ck_data))
    print(len(jafee_data))
    all_images = np.stach((ck_data[0], jafee_data[0]))
    all_labels = np.stach((ck_data[1], jafee_data[1]))
    all_freqim = np.stach((ck_data[2], jafee_data[2]))
    print(ck_data[0].shape)
    print(jafee_data[0].shape)
    print(all_images.shape)
    return

if __name__ == '__main__':
    load_combined(128, 128, 1)
    #TODO: Make sure the class labels are the same
