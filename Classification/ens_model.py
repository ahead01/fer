#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 20:52:20 2019

@author: austin
"""
import os
import tensorflow as tf
import numpy as np

PROJ_DIR = os.environ['PROJ_DIR']
MODEL_DIR = PROJ_DIR + '/Classification/models'

def get_model(height, width, chan, n_cls, h_layers, name):
    inputs = tf.keras.Input(shape=(height, width, chan))
    x = tf.keras.layers.Dense(50)(inputs)
    for _ in range(h_layers):
        x = tf.keras.layers.Dense(50)(x)
    outputs = tf.keras.layers.Dense(n_cls, activation="relu")(x)
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

model = get_ens_model(128, 128, 3, 10)
model.summary()



