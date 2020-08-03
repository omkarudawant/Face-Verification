# Facial Recognition

A simple implementation of facial recognition from images, using Google's Facenet model embeddings.

After cloning the project,


- Download the keras model from **[Here](https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)**


- Install the dependencies with,  `pip install -r requirements.txt`

- To use the REST API, run the `app.py` with `uvicorn app:app` and goto `127.0.0.1:8888/docs` to try the API.

- To verify two images from the standalone script,


    `python facial_recognition.py `
        `--model_path=PATH_TO_MODEL `
        `--image_1=PATH_of_first_image `
        `--image_2=PATH_of_second_image `


- In case you want to verify with other distance metrics (cosine or euclidean with L2 norm),


    `python facial_recognition.py `
        `--model_path=PATH_TO_MODEL `
        `--image_1=PATH_of_first_image `
        `--image_2=PATH_of_second_image `
        `--distance_metric="cosine" OR "euclidean_l2"`
