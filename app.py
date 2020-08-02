from warnings import filterwarnings
filterwarnings('ignore')

import os
import cv2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mtcnn import MTCNN
from keras.models import load_model
from verify_images import FaceVerification

from io import BytesIO
from PIL import Image
from numpy import array

from fastapi import FastAPI, File, UploadFile
import uvicorn

# Initialize app
app = FastAPI()

# Load models
face_detector = MTCNN()

model = load_model('models/facenet_v1.h5')
face_rec = FaceVerification(face_detector, model)


@app.post('/predict')
async def predict(image_1: UploadFile = File(...), image_2: UploadFile = File(...)):
    im1 = await image_1.read()
    im2 = await image_2.read()

    im1 = Image.open(BytesIO(im1))
    im2 = Image.open(BytesIO(im2))

    im1 = array(im1)
    im2 = array(im2)

    print(type(im1), type(im2), '-------\n')

    result = face_rec.extract_and_compare(file1=im1, file2=im2)
    return {'result':'Matched !!!' if result == 1 else 'Not matched...'}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
