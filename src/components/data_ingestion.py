import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from keras.preprocessing.image import ImageDataGenerator

image_width = 150
image_height = 150


@dataclass
class DataIngestionConfig:
    train_path = '../../Dataset/Train'
    val_path = '../../Dataset/Validation'
    test_path = '../../Dataset/Test'


class DataIngestion:

    def __init__(self):
        self.dataIngestionConfig = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered data ingestion method")

        try:
            image_size = (image_width, image_height)

            image_gen = ImageDataGenerator()

            train_images = image_gen.flow_from_directory(
                self.dataIngestionConfig.train_path,
                target_size=image_size,
                class_mode='sparse',
            )

            val_images = image_gen.flow_from_directory(
                self.dataIngestionConfig.val_path,
                target_size=image_size,
                class_mode='sparse',
                shuffle=False,
            )

            test_images = image_gen.flow_from_directory(
                self.dataIngestionConfig.test_path,
                target_size=image_size,
                class_mode='sparse',
                shuffle=False,
            )
            return (
                train_images,
                val_images,
                test_images
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    dataIngestion = DataIngestion()
    dataIngestion.initiate_data_ingestion()
