#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:52:20 2019

@author: austin
"""
import os
import tensorflow as tf

PROJ_DIR = os.environ['PROJ_DIR']
MODEL_DIR = PROJ_DIR + '/Classification/models'

def create_spatial_model(w, h, c, output_nodes):
    '''Model for the spatial image'''
    # This returns a tensor
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='spatial_img')

    # a layer instance is callable on a tensor, and returns a tensor
    x = tf.keras.layers.Conv2D(128, (3,3), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(32,32))(x)
    #x = inputs
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='spatial_HL1')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='spatial_HL2')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='spatial_HL3')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='spatial_HL4')(x)
    outputs = tf.keras.layers.Dense(output_nodes, activation='relu', name='spatial_output')(x)
    model= tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def create_freq_model(w, h, c, output_nodes):
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='freq_img')
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(32, 32))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='freq_HL1')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='freq_HL2')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='freq_HL3')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='freq_HL4')(x)
    outputs = tf.keras.layers.Dense(output_nodes, activation='relu', name='freq_output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def create_deep_model(w, h, c, output_nodes, n_classes):
    spatial_in = create_spatial_model(w, h, c, output_nodes)
    freq_in = create_freq_model(w, h, c, output_nodes)

    inputs = tf.concat([spatial_in.output, freq_in.output], axis=1)

    x = tf.keras.layers.Dense(10, activation='relu', name='HL1')(inputs)
    x = tf.keras.layers.Dense(10, activation='relu', name='HL2')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='HL3')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='HL4')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='HL5')(x)
    output = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = tf.keras.models.Model(inputs = [spatial_in.inputs, freq_in.inputs], outputs = output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_simple_model(w, h, _, n_classes):
    inputs = tf.keras.layers.Input(shape=(w, h), name='img')
    x = tf.keras.layers.Flatten()(inputs)
    if False:
        x = tf.keras.layers.Dense(20, activation='relu', name='HL1')(x)
        x = tf.keras.layers.Dropout(.01)(x)
        x = tf.keras.layers.Dense(20, activation='relu', name='HL2')(x)
        x = tf.keras.layers.Dense(15, activation='relu', name='HL3')(x)
        x = tf.keras.layers.Dense(20, activation='relu', name='HL4')(x)
        x = tf.keras.layers.Dropout(.001)(x)
        x = tf.keras.layers.Dense(10, activation='relu', name='HL5')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL6')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL7')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL8')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL9')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL10')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL11')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL12')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL13')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL14')(x)
        x = tf.keras.layers.Dense(5, activation='relu', name='HL15')(x)
    else:
        x = tf.keras.layers.Dense(100, activation='relu', name='HL1',
        kernel_regularizer= tf.keras.regularizers.l2(l=0.001))(x)

        x = tf.keras.layers.Dense(100, activation='relu', name='HL2',
        kernel_regularizer= tf.keras.regularizers.l2(l=0.001))(x)
        x = tf.keras.layers.Dense(100, activation='relu', name='HL3',
        kernel_regularizer= tf.keras.regularizers.l2(l=0.001))(x)

    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def create_wide_model(w,h, c, _, n_classes):
    inputs = tf.keras.layers.Input(shape=(w, h ), name='img')
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(100, activation='relu', name='HL1')(x)
    x = tf.keras.layers.Dense(200, activation='relu', name='HL2')(x)
    x = tf.keras.layers.Dense(300, activation='relu', name='HL3')(x)
    x = tf.keras.layers.Dense(200, activation='relu', name='HL4')(x)
    x = tf.keras.layers.Dense(100, activation='relu', name='HL5')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def create_conv_model(w, h, c, _, n_classes):
    # This returns a tensor
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='spatial_img')
    #x = inputs

    # a layer instance is callable on a tensor, and returns a tensor
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)

    if True:
        x = tf.keras.layers.MaxPooling2D(data_format='channels_last')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
        x = tf.keras.layers.MaxPooling2D(data_format='channels_last')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(x)
        x = tf.keras.layers.MaxPooling2D(data_format='channels_last')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(x)
        x = tf.keras.layers.MaxPooling2D(data_format='channels_last')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(x)
        x = tf.keras.layers.MaxPooling2D(data_format='channels_last')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='spatial_HL1')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='spatial_HL2')(x)
        x = tf.keras.layers.Dense(1000, activation='relu', name='spatial_HL3')(x)
        x = tf.keras.layers.Dense(100, activation='relu', name='spatial_HL4')(x)

    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='spatial_output')(x)

    model= tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def create_face_model(img_height, img_width, outputs):
    inputs = tf.keras.layers.Input(shape=(img_height, img_width))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    outputs = tf.keras.layers.Dense(outputs, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def face_and_eye_model(img_height, img_width, eye_height, eye_width, n_classes):
    oputput_nodes = 40
    face_model = create_face_model(img_height, img_width, 50)
    eye_model = create_face_model(eye_height, eye_width, 1)

    inputs = tf.concat([face_model.output, face_model.output], axis=1)
    #inputs = face_model.output

    x = tf.keras.layers.Dense(50, activation='relu', name='HL1')(inputs)

    output = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = tf.keras.models.Model(inputs = [face_model.inputs, eye_model.inputs], outputs = output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def hap_model(img_height, img_width, n_classes):
    inputs = tf.keras.layers.Input(shape=(img_height, img_width))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(10, activation='relu',
        kernel_regularizer= tf.keras.regularizers.l2(l=0.05))(x)
    x = tf.keras.layers.Dropout(.03)(x)


    x = tf.keras.layers.Dense(25, activation='relu',
        kernel_regularizer= tf.keras.regularizers.l2(l=0.05))(x)
    x = tf.keras.layers.Dropout(.03)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
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


