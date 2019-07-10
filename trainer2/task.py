
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import h5py
from tensorflow.python.lib.io import file_io
from google.cloud import storage
import argparse
#from PIL import Image
import numpy as np
from . import model
import cv2
import glob
#export GOOGLE_APPLICATION_CREDENTIALS="/path/to/file.json"
from os.path import basename
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
  images = []
  labels = []
  for file in glob.glob(dataset_path+"/*/*.jpg"):
    #label = os.path.abspath(dataset_path+"/*/*.jpg")
    #label = basename(os.getcwd())
    label = os.path.split(os.path.dirname(dataset_path+"/*"+file))[1]


    #os.path.split(os.path.abspath(mydir))[0]
    #label = glob.glob(dataset_path+"/*")
    labels = np.append(labels, label)
    img = cv2.resize(cv2.imread(file), (100,100))
    images = np.append(images, img)
  images = np.array(images).reshape(-1, 100, 100,3)
  #print(labels.shape)
  for n, i in enumerate(labels):
    if i == 'Acura Integra Type R 2001':
      labels[n] = 1
    elif i == 'BMW 1 Series Coupe 2012':
      labels[n] = 0
  #labels = labels.reshape(-1) 
  #print(labels.shape)
  #labels = labels.reshape(-1) 
  #labels = int(labels)  
  return images, labels   
 

#train_data, train_labels = read_images('/Users/tsitsimarote/Desktop/convolutional/Google_Tutorials/keras/stanford-car-dataset-by-classes-folder/car_data/train1')
#print(train_data.dtype, train_labels.dtype)
#print(type(train_data), type(train_labels))
#train_labels = np.asarray(train_labels, dtype = int)
#print(train_data.dtype, train_labels.dtype)
#print(train_labels)
#test_data, test_labels = read_images('/Users/tsitsimarote/Desktop/convolutional/Google_Tutorials/keras/stanford-car-dataset-by-classes-folder/car_data/test1')
#print(train_data.dtype, train_labels.dtype)
#print(type(test_data), type(test_labels))
#test_labels = np.asarray(test_labels, dtype = int)
#print(train_data.dtype, train_labels.dtype)
def train_and_evaluate(args):
  """Helper function: Trains and evaluates model.

  Args:
    hparams: (dict) Command line parameters passed from task.py
  """
  # Loads data.
  train_data, train_labels = read_images('gs://cars_data_tuts/car_data/train')
  test_data, test_labels = read_images('gs://cars_data_tuts/car_data/test')
  #train_data, train_labels = read_images('/Users/tsitsimarote/Desktop/convolutional/Google_Tutorials/keras/stanford-car-dataset-by-classes-folder/car_data/train1')
  #test_data, test_labels = read_images('/Users/tsitsimarote/Desktop/convolutional/Google_Tutorials/keras/stanford-car-dataset-by-classes-folder/car_data/test1')
  run_config = tf.estimator.RunConfig(save_checkpoints_steps=1000)
  #train_steps = args.num_epochs * len(train_data) / args.batch_size
  train_labels = np.asarray(train_labels).astype('float32').reshape((-1, 1))
  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: model.input_fn(
    train_data,
    train_labels,
    args.batch_size,
    mode=tf.estimator.ModeKeys.TRAIN),
    max_steps=10)
  exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
  test_labels = np.asarray(test_labels).astype('float32').reshape((-1, 1))
  eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: model.input_fn(
    test_data,
    test_labels,
    args.batch_size,
    mode=tf.estimator.ModeKeys.EVAL),
    steps=1,
    exporters=[exporter],
    start_delay_secs=10,
    throttle_secs=10)
  estimator = model.keras_estimator(
    	model_dir=args.job_dir,
    	config=run_config,
    	params={'learning_rate': args.learning_rate,'num_features': train_data.shape[1]})
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  #CNN_Model = model.keras_estimator()

  #CNN_Model.fit(train_data, train_labels, steps_per_epoch=2, epochs=1, verbose=1)
  #loss_and_metrics = CNN_Model.evaluate(test_data, test_labels, batch_size=128)
  
if __name__ == '__main__':

  args = get_args()
  tf.logging.set_verbosity(args.verbosity)

  hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(args)