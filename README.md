# Facial Verification

A facial verification API using FastAPI, with Google's Facenet model for embeddings.

After cloning the project,

- In case you want to test the API in docker,
    
    `docker build -t YourFavouriteNameForImage .` and then to run the image in a container,
    
    `docker run -d --name YourFavouriteNameForContainer -p 80:80 YourFavouriteNameForImage`
    
    After which you can try the API at `localhost/docs` or `127.0.0.1/docs`
    
- In case you want try the API without docker,

    - Model is available at **[Here](https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)**

    - Install the dependencies with,  `pip install -r requirements.txt`
    
    - Then switch to `app/` directory by `cd app/`
    
        - To use the REST API, run the `main.py` und with `uvicorn main:app` and goto `localhost:8000/docs` or `127.0.0.1:8000/docs` to try the API.

        - To verify two images from the standalone script,

            `python facial_recognition.py --model_path=PATH_TO_MODEL --image_1=PATH_of_first_image --image_2=PATH_of_second_image`


        - In case you want to verify with other distance metrics (cosine or euclidean with L2 norm),

            `python facial_recognition.py --model_path=PATH_TO_MODEL --image_1=PATH_of_first_image --image_2=PATH_of_second_image --distance_metric="cosine" OR "euclidean_l2"`

Learn more about Facenet ***[Here](https://arxiv.org/abs/1503.03832)***

Learn more about FastAPI ***[Here](https://fastapi.tiangolo.com/)***

Learn more about Docker ***[Here](https://www.docker.com/)***
