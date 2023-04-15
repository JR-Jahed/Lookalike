from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU
from tensorflow import keras
import matplotlib.pyplot as plt


def show_image(image, title):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


model_path = "../../artifacts/My Model"
batch_size = 32


class MyModel:
    def __init__(self, train_images, val_images, test_images, input_shape):

        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images
        self.classes = {v: k for k, v in train_images.class_indices.items()}

        resize_rescale = keras.models.Sequential([
            keras.layers.Resizing(width=input_shape[0], height=input_shape[1], name="resize"),
            keras.layers.Rescaling(1. / 255, name="rescale")
        ])

        self.model = keras.models.Sequential([
            resize_rescale,
            keras.layers.Input(input_shape, name="input"),

            Conv2D(32, 3, name="conv1"),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),

            Conv2D(32, 3, name="conv2"),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),

            Conv2D(64, 3, name="conv3"),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),

            Conv2D(64, 3, name="conv4"),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),

            Conv2D(128, 3, name="conv5"),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),

            Flatten(),

            Dense(256, activation='relu', name="dense1"),
            Dropout(.2),
            Dense(len(self.classes), activation='softmax')
        ])

    def add_layer(self, layer):
        self.model.add(layer)

    def train(self, epochs):

        model_ckpt = keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        )

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        history = self.model.fit(
            self.train_images,
            validation_data=self.val_images,
            epochs=epochs,
            callbacks=[model_ckpt]
        )

        epochs_range = range(1, len(history.history['accuracy']) + 1)

        plt.title('Training and Validation accuracy')
        plt.xticks(epochs_range)
        plt.plot(epochs_range, history.history["accuracy"], 'bo', label='Training acc')
        plt.plot(epochs_range, history.history["val_accuracy"], 'b', label='Validation acc')
        plt.legend()
        plt.show()

    def test(self):
        model = keras.models.load_model(model_path)

        predictions = model.predict(self.test_images)
        predictions = predictions.argmax(axis=-1)

        self.correct_guess(predictions)

        while True:
            num = input("Enter a number between 0 and {}: ".format(self.test_images.samples - 1))
            num = int(num)

            if num == -1:
                break

            elif 0 <= num < self.test_images.samples:

                x = int(num / batch_size)
                y = num % batch_size

                show_image(self.test_images[x][0][y] / 255, "class: {}   prediction: {}".
                           format(self.classes[self.test_images[x][1][y]], self.classes[predictions[num]]))

            else:
                print("Please enter a correct number")

    def correct_guess(self, predictions):

        correct = 0
        incorrect = 0

        for i in range(len(self.test_images)):
            for j in range(len(self.test_images[i][1])):

                idx = i * batch_size + j

                if self.test_images[i][1][j] == predictions[idx]:
                    correct += 1
                else:
                    incorrect += 1

        print('correct: ', correct, 'incorrect: ', incorrect, 'test accuracy: ', (correct * 1.0) / len(predictions))


