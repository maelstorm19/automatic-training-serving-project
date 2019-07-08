import numpy as np
import requests
import base64
import argparse

import tensorflow as tf
from keras.preprocessing import image
import json
ap = argparse.ArgumentParser()
ap.add_argument('--image', help='image path')
args = vars(ap.parse_args())

image_path =args['image']

# Preprocessing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(80, 80))) / 255.

img = img.astype('float16')


payload = {
    "instances": [img.tolist()]
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9000/v1/models/cat-dog:predict', json=payload)
#print(r.content)
pred = json.loads(r.content.decode('utf-8'))

print(json.dumps((pred['predictions'])))


