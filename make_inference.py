import numpy as np
import requests
import tensorflow as tf
from keras.preprocessing import image
import json

image_path ='dataset_repository/test1/1.jpg'

# Preprocessing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(80, 80))) / 255.

# this line is added because of a bug in tf_serving(1.10.0-dev)
img = img.astype('float16')

payload = {
    "instances": [{'input_image': img.tolist()}]
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9000/v1/models/cat-dog:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))

print(json.dumps((pred['predictions']))[0])


