#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:52:20 2019

@author: austin
"""
import tensorflow as tf


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


def create_simple_model(w, h, c, _, n_classes):
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='img')
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(20, activation='relu', name='HL1')(x)
    x = tf.keras.layers.Dense(20, activation='relu', name='HL2')(x)
    x = tf.keras.layers.Dense(20, activation='relu', name='HL3')(x)
    x = tf.keras.layers.Dense(20, activation='relu', name='HL4')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='HL5')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def create_wide_model(w,h, c, _, n_classes):
    inputs = tf.keras.layers.Input(shape=(w, h, c), name='img')
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