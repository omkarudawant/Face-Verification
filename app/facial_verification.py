from warnings import filterwarnings
filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mtcnn import MTCNN
import numpy as np
from PIL import Image
import cv2


class FaceVerification:
    """
    A class for facial recognition with Google's Facenet model embeddings
    """

    def __init__(self, face_detector, model):
        super().__init__()
        self.detector = face_detector
        self.model = model

    def _adjust_gamma(self, image, gamma=2):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0)**invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def extract_face(self,
                     filename,
                     required_size=(160, 160),
                     show_extracted_face=False):
        """
        Extract a single face from a given photograph
        """
        # load image from file
        resized = cv2.cvtColor(filename, cv2.COLOR_BGR2RGB)
        # print(f'Dim Before resizing: {pixels.shape}')

        # Compressing image
        scale_percent = 50  # percent of original size
        width = int(resized.shape[1] * scale_percent / 100)
        height = int(resized.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized = cv2.resize(resized, dim, interpolation=cv2.INTER_AREA)

        # detect faces in the image
        results = self.detector.detect_faces(resized)

        if len(results) > 1:  # image contains more than 1 faces
            sizes = dict()
            for i, res in enumerate(results):
                x1, y1, width, height = res['box']
                area = width * height
                sizes[i] = area
            max_key = [max(sizes, key=sizes.get)][0]
            # Biggest bounding box
            x1, y1, width, height = results[max_key]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = resized[y1:y2, x1:x2]

        elif len(results) == 1:  # image contains exactly 1 face
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = resized[y1:y2, x1:x2]

        else:  # Face not detected because of backlight
            resized = cv2.convertScaleAbs(resized, alpha=1.1, beta=2)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            resized = resized[30::]

            # detect faces in the image
            results = self.detector.detect_faces(resized)

            if len(results) < 1:
                face = np.random.rand(160, 160, 3) * 255
            else:
                # extract the bounding box from the first face
                x1, y1, width, height = results[0]['box']
                x2, y2 = x1 + width, y1 + height

                # extract the face
                face = resized[y1:y2, x1:x2]

        # resize pixels to the model size
        face_array = cv2.resize(face,
                                required_size,
                                interpolation=cv2.INTER_LINEAR_EXACT)
        if show_extracted_face:
            cv2.imshow('face', face_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        face_array = np.expand_dims(face_array, axis=0)
        face_array = face_array / 255
        return face_array

    def compare_distance(self,
                         embedding1=0.0,
                         embedding2=0.0,
                         thresholds=None,
                         distance_metric=None):

        cosine_threshold, euclidean_threshold, euclidean_l2_threshold = thresholds
        # calculate distance with given metric
        if distance_metric == 'euclidean':
            euclidean_distance = round(
                np.linalg.norm(embedding1 - embedding2), 4)
            if euclidean_distance <= euclidean_threshold:
                return 1
            else:
                return 0

        elif distance_metric == 'cosine':
            cosine_distance = (embedding1 @ embedding2.T) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            if cosine_distance <= cosine_threshold:
                return 1
            else:
                return 0

        elif distance_metric == 'euclidean_l2':
            norm1 = embedding1 / np.sqrt(
                np.sum(np.multiply(embedding1, embedding1)))
            norm2 = embedding2 / np.sqrt(
                np.sum(np.multiply(embedding2, embedding2)))
            euclidean_l2_distance = np.linalg.norm(norm1 - norm2)
            if euclidean_l2_distance <= euclidean_l2_threshold:
                return 1
            else:
                return 0

    def extract_and_compare(self,
                            file1=None,
                            file2=None,
                            model=None,
                            thresholds=(0.4, 11.5, 0.8),
                            distance_metric='euclidean'):
        # extract faces
        img1_face = self.extract_face(filename=file1)
        img2_face = self.extract_face(filename=file2)

        # Detect embeddings
        embd1 = self.model.predict(img1_face)
        embd2 = self.model.predict(img2_face)

        # Compute distances and verify
        result = self.compare_distance(embedding1=embd1,
                                       embedding2=embd2,
                                       thresholds=thresholds,
                                       distance_metric=distance_metric)
        return result


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        'Script for facial recognition threshold testing\n')
    arg_parser.add_argument('--model_path',
                            help='Str, Path to the Model for calculating facial embeddings',
                            required=True)
    arg_parser.add_argument(
        '--image_1', help='Str, Path of the first image', required=True)
    arg_parser.add_argument(
        '--image_2', help='Str, Path of the second image', required=True)
    arg_parser.add_argument(
        '--distance_metric', help='Str, Distance metric to use for verification of images, '
        'could be of the following, "cosine", "euclidean"(default), "euclidean_l2" ')

    args = arg_parser.parse_args()

    # Load face detector
    detector = MTCNN()

    # Load model
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(args.model_path, compile=False)

    face_rec = FaceVerification(face_detector=detector, model=model)
    image_1 = cv2.imread(args.image_1)
    image_2 = cv2.imread(args.image_2)
    result = face_rec.extract_and_compare(
        file1=image_1, file2=image_2)
    print('Matched' if result == 1 else "Not matched")
