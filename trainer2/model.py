# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D, Lambda, Convolution2D 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras import backend as K
#K.set_image_dim_ordering('th')
tf.logging.set_verbosity(tf.logging.INFO)


def keras_estimator(model_dir, config, params):
  """Creates a Keras Sequential model with layers.

  Args:
    model_dir: (str) file path where training files will be written.
    config: (tf.estimator.RunConfig) Configuration options to save model.
    learning_rate: (int) Learning rate.

  Returns:
    A keras.Model
  """
  num_filters = 32            # No. of conv filters
  max_pool_size = (2,2)       # shape of max_pool
  conv_kernel_size = (3, 3)    # conv kernel shape
  imag_shape = (100, 100, 3)
  num_classes = 2
  drop_prob = 0.5
  model = Sequential()
# 1st Layer
  model.add(Convolution2D(num_filters, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
  model.add(MaxPooling2D(pool_size=max_pool_size))
# 2nd Convolution Layer
  model.add(Convolution2D(num_filters*2, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
  model.add(MaxPooling2D(pool_size=max_pool_size))
# 3nd Convolution Layer
  #model.add(Convolution2D(num_filters*4, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
  #model.add(MaxPooling2D(pool_size=max_pool_size))
#Fully Connected Layer
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))   #Fully connected layer
# Dropout some neurons to reduce overfitting
  model.add(Dropout(drop_prob))
#Readout Layer
  model.add(Dense(num_classes, activation='sigmoid'))
  #model = Sequential()
  #model.add(Dense(2, activation='relu', input_shape=(100,100,3)))
  #model.add(Dense(64, activation='relu'))
  #model.add(Dropout(0.2))
    # Already overfitting, no need to add this extra layer
    # model.add(Dense(layer1_size, activation='relu'))
    # model.add(Dropout(0.2))
  #model.add(Dense(n, activation='softmax'))

  #model.add(Flatten(input_shape=(28, 28)))
  #model.add(Dense(128, activation=tf.nn.relu))
  #model.add(Dense(10, activation=tf.nn.softmax))
  #model.add(Conv2D(32, kernel_size=(8,8), strides=(2,2), activation='relu', input_shape=(128,128,3)))
  #model.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
  #model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  #model.add(MaxPooling2D(pool_size=(2, 2)))
  #model.add(Dropout(0.25))
  #model.add(Flatten())
  #model.add(Dense(128, activation='relu'))
  #model.add(Dropout(0.5))
  #model.add(Dense(2, activation='softmax'))
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  #return model
  return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir, config=config)

def input_fn(features, labels, batch_size, mode):
	if labels is None:
		inputs = features
	else:
		inputs = (features, labels)
	dataset = tf.data.Dataset.from_tensor_slices(inputs)
	if mode == tf.estimator.ModeKeys.TRAIN:
		dataset = dataset.shuffle(1000).repeat().batch(batch_size)
	if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
		dataset = dataset.batch(batch_size)
	return dataset.make_one_shot_iterator().get_next()

def serving_input_fn():
	feature_placeholder = tf.placeholder(tf.float32, [None, 100, 100, 3])
	features = feature_placeholder
	return tf.estimator.export.TensorServingInputReceiver(features, feature_placeholder)












