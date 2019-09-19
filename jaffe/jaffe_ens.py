import jaffe_fer
import os
import re
import cv2
import tensorflow as tf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pre_processing
import jaffe
import model_gen

# Globals
label_names = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

DATA_DIR = os.environ['DATA_DIR']
PROJECT_DIR = os.environ['PROJ_DIR']

if __name__ == '__main__':
    img_height = 256
    img_width = 256
    eye_height = 128
    eye_width = 64

    images, labels, eye_imgs = jaffe_fer.load_images(img_height, img_width, 1, edge_images=True)

    n_classes =len(np.unique(labels))
    print(f'There are {n_classes} unique classes.')

    neutral_model = model_gen.load_trained_model('jaffe_neu_002')
    for i, layer in enumerate(neutral_model.layers):
        layer._name = 'neutral_' + str(i)

    anger_model   = model_gen.load_trained_model('jaffe_ang_002')
    for i, layer in enumerate(anger_model.layers):
        layer._name = 'anger_' + str(i)

    disgust_model = model_gen.load_trained_model('jaffe_dis_002')
    for i, layer in enumerate(disgust_model.layers):
        layer._name = 'disgust_' + str(i)

    fear_model    = model_gen.load_trained_model('jaffe_fea_001')
    for i, layer in enumerate(fear_model.layers):
        layer._name = 'fear_' + str(i)

    happy_model   = model_gen.load_trained_model('jaffe_happy_001')
    for i, layer in enumerate(happy_model.layers):
        layer._name = 'happy_' + str(i)

    sadness_model = model_gen.load_trained_model('jaffe_sad_002')
    for i, layer in enumerate(sadness_model.layers):
        layer._name = 'sad_' + str(i)

    surprise_model = model_gen.load_trained_model('jaffe_sup_002')
    for i, layer in enumerate(surprise_model.layers):
        layer._name = 'surprise_' + str(i)

    inputs = tf.concat([neutral_model.output,
                        anger_model.output,
                        disgust_model.output,
                        fear_model.output,
                        happy_model.output,
                        sadness_model.output,
                        surprise_model.output,
    ], axis=1)
    x = tf.keras.layers.Dense(50, activation='relu', name='HL1')(inputs)
    output = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    print(anger_model['anger_0'])
    exit()
    #print(anger_model.__dir__())
    #print(anger_model.__dict__)
    for layer in anger_model.layers:
        print(layer.name)

        break
    exit()

    model = tf.keras.models.Model(inputs = [neutral_model.inputs,
                        anger_model.inputs,
                        disgust_model.inputs,
                        fear_model.inputs,
                        happy_model.inputs,
                        sadness_model.inputs,
                        surprise_model.inputs,], outputs = output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

        # Define the Keras TensorBoard callback.
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    # Train the model.
    model.fit(
        [images, images, images, images, images, images, images],
        labels,
        batch_size=20,
        epochs=100,
        callbacks=[tensorboard_callback], validation_split=0.2)


