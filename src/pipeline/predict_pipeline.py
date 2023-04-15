import keras
import numpy as np
import sys
from src.exception import CustomException
from PIL import Image
from io import BytesIO

model_path = "../../artifacts/My Model"

class_names = ['Amir Khan', 'Angelina Jolie', 'Brad Pitt', 'Denzel Washington', 'Hugh Jackman',
               'Jennifer Lawrence', 'Johnny Depp', 'Kate Winslet', 'Leonardo DiCaprio', 'Megan Fox',
               'Natalie Portman', 'Nicole Kidman', 'Robert Downey Jr', 'Salman Khan', 'Sandra Bullock',
               'Shahrukh Khan', 'Tom Cruise', 'Tom Hanks', 'Will Smith']


class PredictPipeline:

    def predict(self, image):

        try:

            model = keras.models.load_model(model_path)

            prediction = model.predict(image)

            predicted_class = class_names[np.argmax(prediction[0])]
            confidence = np.max(prediction[0])

            return {
                'class': predicted_class,
                'confidence': float(confidence)
            }

        except Exception as e:
            raise CustomException(e, sys)


class Data:
    def __init__(self, data):
        image = np.array(Image.open(BytesIO(data)))

        self.image_batch = np.expand_dims(image, 0)
















