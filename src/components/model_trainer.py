from src.my_model import MyModel


class ModelTrainer:
    def initiate_model_trainer(self, train_images, val_images, test_images, input_shape, training=True):

        model = MyModel(
            train_images=train_images,
            val_images=val_images,
            test_images=test_images,
            input_shape=input_shape
        )

        if training:
            model.train(15)
            model.test()
        else:
            model.test()
