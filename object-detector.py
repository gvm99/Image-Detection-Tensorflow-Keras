"""
Created on Wed Aug  8 23:07:00 2018

@author: guivm
"""
from os.path import join

image_dir = 'C:/Users/guivm/Downloads/inputs'
img_paths = [join(image_dir, filename) for filename in 
                           ['perro-husky-siberiano.jpg',
                            'beagle.jpg','monitor.jpg']]

import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import json

image_size = 224
def decode_predictions(preds, top=5, class_list_path=None):
  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  CLASS_INDEX = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)


from tensorflow.python.keras.applications import ResNet50

my_model = ResNet50(weights='resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

import sys
# Add directory holding utility functions to path to allow importing
from IPython.display import Image, display

most_likely_labels = decode_predictions(preds, top=3, class_list_path='resnet50/imagenet_class_index.json')
likely = most_likely_labels[0][0][1]
for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print("A resposta é :"+most_likely_labels[i][0][1])
