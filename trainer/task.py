
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import h5py
from tensorflow.python.lib.io import file_io

import argparse
#from PIL import Image
import numpy as np
from . import model
import cv2
import glob

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_args():
  """Argument parser.

	Returns:
	  Dictionary of arguments.
	"""
  
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--job-dir',
    type=str,
    required=True,
    #default='gs://cars_data_tuts/model1',
    help='GCS location to write checkpoints and export models')
  parser.add_argument(
    '--num-epochs',
    type=float,
    default=5,
    help='number of times to go through the data, default=5')
  parser.add_argument(
    '--batch-size',
    default=128,
    type=int,
    help='number of records to read during each training step, default=128')
  parser.add_argument(
    '--learning-rate',
    default=.01,
    type=float,
    help='learning rate for gradient descent, default=.001')
  parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')
  return parser.parse_args()

def read_images(dataset_path):
  #images=np.empty([100, 1])
  #labels=np.empty([1,1])
  images = []
  labels = []
  for root, dirs, files in os.walk(dataset_path):
    for name in dirs:
      p = os.path.join(root, name)
      os.chdir(p)
      for file in glob.glob(p+"/*.jpg"):
        labels = np.append(labels, name)
        img = cv2.resize(cv2.imread(file), (100,100))
          #img = cv2.resize(img, (100,100))
          #print(img.shape)
        images = np.append(images, img)
          #images = images.reshape(-1, 100, 100)
          #print(images.shape)
          #count = count + 1
          #print(count, thedir)
          #labels = np.append(labels, thedir)
          
          
        #images = np.array([cv2.imread(file) for file in glob.glob(p+"/*.jpg")])
        #images = images.reshape(len(images), 224, 224)
        #images = np.array(images)
  images = images.reshape(-1, 100, 100, 3)
#print(len(labels))
#labels = le.fit(np.array(labels))
  for n, i in enumerate(labels):
    if i == 'Acura Integra Type R 2001':
      labels[n] = 1
    elif i == 'BMW 1 Series Coupe 2012':
      labels[n] = 0
  #print(x.shape)
  labels = labels.reshape(len(labels), 1)    
  return images, labels        

train_data, train_labels = read_images('/Users/tsitsimarote/Desktop/convolutional/Google_Tutorials/keras/stanford-car-dataset-by-classes-folder/car_data/train1')
print(train_data.shape, train_labels.shape)
test_data, test_labels = read_images('/Users/tsitsimarote/Desktop/convolutional/Google_Tutorials/keras/stanford-car-dataset-by-classes-folder/car_data/test1')
print(test_data.shape, test_labels.shape)
def train_and_evaluate(hparams):
  """Helper function: Trains and evaluates model.

  Args:
    hparams: (dict) Command line parameters passed from task.py
  """
  # Loads data.
  CNN_Model = model.keras_estimator()

  CNN_Model.fit(train_data, train_labels, steps_per_epoch=2, epochs=100, verbose=1)
  #loss_and_metrics = CNN_Model.evaluate(test_data, test_labels, batch_size=128)
  
if __name__ == '__main__':

  args = get_args()
  tf.logging.set_verbosity(args.verbosity)

  hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hparams)