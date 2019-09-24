import os
import re
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pre_processing
import jaffe
import model_gen

# Globals


DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']
MODEL_DIR = PROJECT_DIR + '/Classification/models'

CASCADE_DIR = PROJECT_DIR + '/FaceDetection/xml'
FACE_XML = CASCADE_DIR + '/haarcascade_frontalface_default.xml'
EYE_XML = CASCADE_DIR + '/haarcascade_eye.xml'


FACE_CASCADE = cv2.CascadeClassifier(FACE_XML)
EYE_CASCADE = cv2.CascadeClassifier(EYE_XML)

def select_eyes(eyes):
    if len(eyes) == 0:
        return None
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
        return eyes

    if len(eyes) < 2:
        return None

def load_images(img_height, img_width, img_chan, edge_images=False):
    ''' height, width, channels'''
    img_dir = DATA_DIR + '/jaffe/'
    tiff_pattern = re.compile('\.tiff', re.IGNORECASE)
    patterns = jaffe.get_patterns()

    # Count files in dir so I can pre allocate np arrays
    image_count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            image_count += 1

    # Allocate arrays
    labels = np.empty((208), dtype=np.uint8)
    images = np.empty((208, img_height, img_width), dtype=np.uint8)
    eye_imgs = np.zeros((208, 128, 64), dtype=np.uint8)
    count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            # Load image unchanged
            img = cv2.imread(img_dir + file_name, cv2.IMREAD_UNCHANGED)
            faces = FACE_CASCADE.detectMultiScale(img, 1.3, 5)
            for (x,y,w,h) in faces:
                face_img = img[y:y+h, x:x+w]
            img = cv2.resize(face_img, (img_height, img_width))
            eyes = EYE_CASCADE.detectMultiScale(face_img)
            eyes = select_eyes(eyes)
            if eyes is None:
                continue
            tmp_eyes = []
            for (ex,ey,ew,eh) in eyes:
                eye_img = face_img[ey:ey+eh, ex:ex+ew]
                eye_img = cv2.resize(eye_img, (64, 64))
                tmp_eyes.append(eye_img)
            eye_img = np.concatenate((tmp_eyes[0], tmp_eyes[1]), axis=0)

            eye_imgs[count] = eye_img
            # Detect edges
            if edge_images:
                img = pre_processing.edge_detection(img)
            images[count] = img
            for label, pattern in enumerate(patterns):
                if pattern.search(file_name):
                    labels[count] = label
                    count += 1
                    break
    print(count)

    return images, labels, eye_imgs

def load_select_images(img_height, img_width, img_chan, pattern, edge_images=False):
    ''' load images that match the pattern as calss 1 and the others as class 0'''
    img_dir = DATA_DIR + '/jaffe/'
    tiff_pattern = re.compile('\.tiff', re.IGNORECASE)
    # Count files in dir so I can preallocate np arrays
    image_count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            image_count += 1
    # Allocate arrays
    labels = np.empty((208), dtype=np.uint8)
    images = np.empty((208, img_height, img_width), dtype=np.uint8)
    eye_imgs = np.zeros((208, 128, 64), dtype=np.uint8)
    count = 0
    for file_name in os.listdir(img_dir):
        if tiff_pattern.search(file_name):
            # Load image unchanged
            img = cv2.imread(img_dir + file_name, cv2.IMREAD_UNCHANGED)
            faces = FACE_CASCADE.detectMultiScale(img, 1.3, 5)
            for (x,y,w,h) in faces:
                face_img = img[y:y+h, x:x+w]
            img = cv2.resize(face_img, (img_height, img_width))
            eyes = EYE_CASCADE.detectMultiScale(face_img)
            eyes = select_eyes(eyes)
            if eyes is None:
                continue
            tmp_eyes = []
            for (ex,ey,ew,eh) in eyes:
                eye_img = face_img[ey:ey+eh, ex:ex+ew]
                eye_img = cv2.resize(eye_img, (64, 64))
                tmp_eyes.append(eye_img)
            eye_img = np.concatenate((tmp_eyes[0], tmp_eyes[1]), axis=0)

            eye_imgs[count] = eye_img
            # Detect edges
            if edge_images:
                img = pre_processing.edge_detection(img)
            images[count] = img
            if pattern.search(file_name):
                labels[count] = 1
            else:
                labels[count] = 0
            count += 1

    print(count)

    return images, labels, eye_imgs

def get_model(height, width, chan, n_cls, h_layers, name):
    inputs = tf.keras.Input(shape=(height, width, chan), name=f'{name}_inputs')
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", name=f'{name}_conv2d')(inputs)
    x = tf.keras.layers.MaxPooling2D(data_format='channels_last', name=f'{name}_maxpool')(x)
    x = tf.keras.layers.Flatten( name=f'{name}_flatten')(x)
    for layer_num in range(h_layers):
        x = tf.keras.layers.Dense(50, name=f'{name}_d_{layer_num}')(x)
    outputs = tf.keras.layers.Dense(n_cls, activation="relu", name=f'{name}_outputs')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def get_ens_model(height, width, chan, n_cls):
    inputs = tf.keras.Input(shape=(height, width, chan))
    sub_model_1 = get_model(height, width, chan, n_cls, 2, 'sub_model_1')
    sub_model_2 = get_model(height, width, chan, n_cls, 2, 'sub_model_2')
    sub_model_3 = get_model(height, width, chan, n_cls, 2, 'sub_model_3')
    y1 = sub_model_1(inputs)
    y2 = sub_model_2(inputs)
    y3 = sub_model_3(inputs)
    outputs = tf.keras.layers.average([y1, y2, y3])
    ens_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return ens_model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


def save_trained_model(model, model_name):
    'Saves a model in the models folder and returns its path'
    model_path = MODEL_DIR + '/' + model_name + '.h5'
    print('Saving Model:', model_path)
    model.save(model_path)
    return model_path

def load_trained_model(model_name):
    'Load a pre-trained model'
    model_path = MODEL_DIR + '/' + model_name + '.h5'
    print('Loading Model:', model_path)
    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        print('Not a valid model!')
        return None

def visualize_model(model):
    model.summary()

def create_sub_models(height, width, chan):
    models = []

    patterns = [re.compile('HA')
                ,re.compile('SA')
                ,re.compile('SU')
                ,re.compile('AN')
                ,re.compile('DI')
                ,re.compile('FE')
                ,re.compile('NE')]

    label_names = ['happy', 'sadness', 'surprise', 'anger', 'disgust', 'fear', 'neutral']
    # Train Sub-models
    for idx, pattern in enumerate(patterns):
        images, labels, eye_imgs = load_select_images(height, width, chan, pattern, edge_images=False)
        print("Emotion:", label_names[idx])
        print('\timages shape', images.shape)
        print('\tlabels shape', labels.shape)
        print('\teye images shape', eye_imgs.shape)
        print('\tunique labels', np.unique(labels))

        images = images / 255.0 # Reduce range 0:1
        images = images[..., tf.newaxis] # Add an axis

        positives = images[labels == 1]
        negatives = images[labels == 0]
        p_labels = labels[labels == 1]
        n_labels = labels[labels == 0]
        #pre_processing.show_image(positives[0], label_names[idx])

        name = label_names[idx] + '_model'
        h_layers = 1
        n_cls = len(np.unique(labels))
        model = get_model(height, width, chan, n_cls, h_layers, name)
        x = 'y'
        if x == 'n':
            break
        model = compile_model(model)
        model.fit(images, labels,
        epochs=10,
        validation_split=0.2)
        save = 'y'
        if save == 'y':
            for layer in model.layers:
                layer.trainable=False
            save_trained_model(model, name)
            models.append(name)
    return models

def create_ens_model(models, height, width, chan, n_cls):
    trained_models = []
    trained_inputs = []
    trained_outputs = []
    for model_name in models:
        tmp_model = load_trained_model(model_name)
        tmp_model.trainable = False
        trained_models.append(tmp_model)
        trained_inputs.append(tmp_model.inputs)
        trained_outputs.append(tmp_model.outputs)


    inputs = tf.concat(trained_outputs, axis=1)
    x = tf.keras.layers.Dense(50, name='arb_1')(inputs)
    x = tf.keras.layers.Dense(50, name='arb_2')(x)
    x = tf.keras.layers.Dense(50, name='arb_3')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    height = 128
    width = 128
    chan = 1
    images, labels, eye_images = load_images(height, width, chan, edge_images=False)
    n_cls = np.unique(labels)
    print('Number of Classes', n_cls)
    trained_model_names = create_sub_models(height, width, chan)
    print(trained_model_names)
    model = create_ens_model(trained_model_names, height, width, chan, n_cls)
    model = compile_model(model)


    images = images / 255.0 # Reduce range 0:1
    images = images[..., tf.newaxis] # Add an axis

    model.fit(images, labels, epochs=10, validation_split=0.2)