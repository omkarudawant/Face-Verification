from warnings import filterwarnings
filterwarnings('ignore')

import os
import cv2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mtcnn import MTCNN
from facial_verification import FaceVerification

from io import BytesIO
from PIL import Image
from numpy import array

from urllib.request import urlretrieve
from fastapi import FastAPI, File, UploadFile
import uvicorn


# Initialize app
app = FastAPI()

# Load models
face_detector = MTCNN()

s = time.time()
model = tf.keras.models.load_model('models/facenet_v1.h5')
print(f'Took: {(time.time() - s)} s')

face_rec = FaceVerification(face_detector, model)


@app.post('/predict_from_url')
async def predict_from_url(image_1: str, image_2: str):
    file_1 = 'image_1_'+image_1.split('/')[-1]
    file_2 = 'image_2_'+image_2.split('/')[-1]

    urlretrieve(image_1, file_1)
    urlretrieve(image_2, file_2)

    im1 = cv2.imread(file_1)
    im2 = cv2.imread(file_2)

    print(type(im1), type(im2), '\n-------\n')

    s = time.time() 
    result = face_rec.extract_and_compare(file1=im1, file2=im2)
    e = time.time()

    total = round((e - s) * 1000, 2)
    os.remove(file_1)
    os.remove(file_2)
    return {'result': 'Matched !!!' if result == 1 else 'Not matched...', 
    'prediction_time(ms)': total}


@app.post('/predict')
async def predict(image_1: UploadFile = File(...), image_2: UploadFile = File(...)):
    im1 = await image_1.read()
    im2 = await image_2.read()

    im1 = Image.open(BytesIO(im1))
    im2 = Image.open(BytesIO(im2))

    im1 = array(im1)
    im2 = array(im2)

    print(type(im1), type(im2), '\n-------\n')

    s = time.time()
    result = face_rec.extract_and_compare(file1=im1, file2=im2)
    e = time.time()

    total = round((e - s) * 1000, 2)
    return {'result': 'Matched !!!' if result == 1 else 'Not matched...', 
    'prediction_time(ms)': total}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
