"""
Mirza N Ahmed
DS3500 / WintourWardrobe
Date Created: 4/7/2023 / Date Last Updated: 4/10/2023

File name: ml_model
Desc: Trains and stores the Machine Learning model using the Fashion MNIST dataset
"""

# Imports

# file/image handing
import os
import numpy as np
from PIL import Image, ImageOps

# machine learning
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# class for handling the recommender model (loading/creating/training/using)
class FashionRecommender:

    # initialize
    def __init__(self, model_path='model.h5'):
        # keras stores structured data in a format called .h5
        self.model_path = model_path

        # check whether the model exists or not (to save time)
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            self._create_feature_model()
        else:
            self.model = None
            self.feature_model = None

    def _create_feature_model(self):
        """
        Creates a new Keras model that takes in the input of the original ML model
        and outputs the tensor of the 2nd to last layer of the original model.

        The `feature_model` attribute is set to the new instance.
        """
        self.feature_model = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

    def preprocess_data(self):
        """
        Loads and cleans the data
        :return: two tuples containing the pre-processed data
        """
        # load the data, split it into training and testing set
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # scale the pixel values between 0-1
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # one-hot encoding
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return (x_train, y_train), (x_test, y_test)

    def create_model(self):
        """
        Creates Convolutional Neural Network model using Keras Sequential API.
        Updates the model and sets the feature model.
        """

        # define the sequential process
        self.model = Sequential([
            # 2D convolutional layer: 32 filters, 3x3 kernel, ReLU activation
            # shape of each Fashion MNIST image after processing is (28, 28, 1)
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            # 2D max pooling of size 2x2
            MaxPooling2D((2, 2)),
            # another 2D convolutional layer: 64 filters, 3x3 kernel, ReLU activation
            Conv2D(64, (3, 3), activation='relu'),
            # 2D max pooling of size 2x2
            MaxPooling2D((2, 2)),
            # convert output of previous layer to 1D vector
            Flatten(),
            # dense layer: 128 neurons, ReLU activation
            Dense(128, activation='relu'),
            # final dense layer: 10 neurons (one for each class) + softmax activation function (prob. distribution)
            Dense(10, activation='softmax')
        ])

        # compiler: Adam optimizer, Category Cross Entropy loss function (good for multiclass), Accuracy eval metric
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self._create_feature_model()

    def train(self, x_train, y_train, x_test, y_test, epochs=10):
        """
        Trains the ML model using the pre-processed data
        :param x_train: pre-processed training features
        :param y_train: pre-processed training target
        :param x_test: pre-processed testing features
        :param y_test:pre-processed testing features
        :param epochs: number of iterations over training set, default = 10 (arbitrary value, decent training time),
                       large value might overfit
        :return: history: (object) contains info about the training process (loss, accuracy at each epoch)
        """

        # creates model if it does not already exist
        if self.model is None:
            self.create_model()

        # fit performs forward and back propagation to update the weights
        history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        self.model.save(self.model_path) # saves the model to the specified path
        return history



    def predict(self, image_path):
        """
        Predicts the class based on the image
        :param image_path: (string) path of the image file
        :return: (tuple) predicted class and label
        """

        # raise exception if model does not exist
        if self.model is None:
            raise Exception("Model not found. Train or load a model before making predictions.")

        # Using Python Imaging Library
        image = Image.open(image_path).convert('L')  # open image
        image = ImageOps.invert(image)  # invert the color of the image # generally white bg # expects black bg
        image = image.convert('L')  # convert to grayscale
        image = image.resize((28, 28))
        image = np.array(image) / 255.0  # convert image to numpy array + scales value from 0 to 1
        image = np.expand_dims(image, axis=2) # add third dimension (for color channel)
        image = np.expand_dims(image, axis=0) # fourth dimension (batch size)

        prediction = self.model.predict(image) # obtains a probability distribution
        predicted_class = np.argmax(prediction) # obtains index of highest predicted probability

        # Map the predicted label number to the corresponding label title
        label_map = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }
        predicted_title = label_map[int(predicted_class)]

        return predicted_class, predicted_title

# Other functions required for the dashboard
def get_sample_images(class_index, num_samples=9, offset=0):
    """
    Provides set of sample images of a chosen class
    :param class_index: (Int) index of class
    :param num_samples: (Int) number of samples to return
    :param offset: (Int) offset to start getting the samples
    :return: (NumPy Array) samples images
    """
    (x_train, y_train), (_, _) = fashion_mnist.load_data()  # loading all data, only using training data for now
    class_indices = np.where(y_train == class_index)[0][offset:offset+num_samples]  # first n samples w/ applied offset

    # todo: is someone clicks on more, show the next 9 samples
    sample_images = x_train[class_indices]
    return sample_images, len(sample_images)

# I have removed the main from this file since this will be imported and used within dash_file.py
## check during OH todo: (Should I move the part below inside the main of the other file?)
recommender = FashionRecommender()

if not os.path.exists(recommender.model_path):
    (x_train, y_train), (x_test, y_test) = recommender.preprocess_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    recommender.train(x_train, y_train, x_test, y_test)


"""
# TESTING FOR DEBUGGING PURPOSES
if __name__ == '__main__':
    recommender = FashionRecommender()


    if not os.path.exists(recommender.model_path):
        (x_train, y_train), (x_test, y_test) = recommender.preprocess_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        recommender.train(x_train, y_train, x_test, y_test)

    # image_path = 'path/to/your/image.jpg'
    image_path = 'ankleboot3.png'
    # image_path_list = ['ankleboot3.png', 'ankleboot5.png',
    #                    'bag1.png', 'bag2.png',
    #                    'dress1.jpg', 'dress2.jpg', 'dress3.jpg',
    #                    'pullover2.png', 'pullover3.png', 'pullover1.png',
    #                    'sample_image.png',
    #                    'shirt1.png',
    #                    'sneaker3.png', 'sneaker4.jpg',
    #                    'trouser1.png', 'trouser2.png',
    #                    'tshirt1.jpg', 'tshirt2.jpg', 'tshirt3.png',
    #                    'coat1.png', 'coat1.png',
    #                    'sandal1.png']

    print("START....")
    # for image_path in image_path_list:
    #     print(image_path)
    #     predicted_class, predicted_title = recommender.predict(image_path)
    #     print(f"Predicted class: {predicted_class}")
    #     print(f"Predicted title: {predicted_title}")
    #     print("---")

    print("END.....")
"""