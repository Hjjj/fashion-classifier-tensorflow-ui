import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import os

TRAINING_EPOCHS = 5
MODEL_PATH = "models/fashion-ai-model.keras"
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class AIFashionImageRecognitionModel:
    def __init__(self, status_label=None):
        self.status_label = status_label
        self.index = 0  # This var keeps state of the current image being processed
        self.load_fashion_data()  # Load fashion MNIST images with labels
        
        # Try to load the on-disk model, if not found, retrain the model
        if not self.open_serialized_model():
            self.status("Model not found, retraining...")
            self.retrain(TRAINING_EPOCHS)
        
        # Evaluate the model on the test data to see how well it works
        self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print(f"Pre-trained model accuracy: {self.accuracy * 100:.2f}%")
        print(f"Pre-trained model loss: {self.loss:.4f}")

        # Save the model to disk
        self.save_serialized_model()

    def load_fashion_data(self):
        # Load the Fashion MNIST dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        # Normalize the data to range between 0 and 1
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        
        # Reshape the data to add a channel dimension (required for Conv2D layers)
        # The new channel dimension represents the one color in a gray scale image
        # if this were a color image (RGB), the channel dimension would be 3
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28, 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28, 1))

    def open_serialized_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.status(f"Model loading from {MODEL_PATH}")
                # Load the pre-trained model
                self.model = tf.keras.models.load_model(MODEL_PATH)
                
                # Compile the model after loading
                self.model.compile(optimizer='adam',
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                   metrics=['accuracy'])
                return True
            except OSError as e:
                self.status(f"Error loading model: {e}")
                self.model = None
        return False

    def save_serialized_model(self):
        # Need a folder to save the model in. 
        if not os.path.exists("models"):
            self.status("Creating models folder")
            os.makedirs("models")
        
        self.status("Saving the model")
        self.model.save(MODEL_PATH)

    def status(self, message):
        # Update the status label on the ui form
        if self.status_label:
            self.status_label.config(text=message)
        print(message)

    def recognize_image(self, image_array):
        # Preprocess the image by adding a batch dimension
        """
        This line of code uses np.expand_dims to add an extra 
        dimension to the image_array. The axis=0 parameter 
        specifies that the new dimension should be added at the 
        beginning, effectively converting a single image of 
        shape (28, 28, 1) to a batch of one image with 
        shape (1, 28, 28, 1).
        Why is This Process Needed for model.predict?
        The model.predict method in Keras expects the input data 
        to have the shape (batch_size, height, width, channels). 
        Even if you are predicting the class for a single image, 
        you need to provide it as a batch of one image.
        """
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict the class of the image
        # This is the query to the model to predict the class of the image.
        predictions = self.model.predict(image_array)
        # The line below gets the answer that the model returned back.
        # One of the nodes in the output layer will have the highest 
        # value. There is a node for each of the clothing types, 
        # so the node with the highest value is the is the clothing 
        # type that the model thinks the image is.
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]  # Convert to string label
        
        return predicted_label

    def interpret_next_image(self):
        # this function queries the model to predict the class of the next image in the test set.
        # so we are looping through test data and asking the ai 
        # to predict the type of each image in the test data.
        if self.index < len(self.x_test):
            image_array = self.x_test[self.index]
            self.index += 1
            
            # getting a clothing image from the test data
            # Convert the image array to a PIL image for display
            image = Image.fromarray((np.squeeze(image_array) * 255).astype(np.uint8))
            image_tk = ImageTk.PhotoImage(image)
            
            # Recognize the image with ai, and put the name of the 
            # clothing type to the label in the ui.
            predicted_label = self.recognize_image(image_array)
            return image_tk, predicted_label
        return None, None

    def retrain(self, epochs):
        # Create the CNN model
        self.status("Creating the CNN model")

        # This is where the model is defined and created. 
        # The model is a Convolutional Neural Network (CNN).
        # each of the lines below defines a layer in the model.
        """
        Purpose of the Entire Model
        Input: Grayscale images of size 28x28
        Processing:
        Convolutional layers extract hierarchical features from the images.
        Max-pooling layers reduce spatial dimensions and emphasize the most important features.
        Dense layers perform classification based on these features.
        Output: A probability distribution over 10 classes, used for predicting the class of the input image.
        This structure is ideal for image classification tasks like handwritten digit recognition (e.g., MNIST dataset).

        Description of Each Layer:

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        Extracts features such as edges, corners, or textures from the input image.
        Applies 32 convolutional filters of size 3x3
        3x3 to the input image.
        Each filter slides across the input to produce feature maps.
        Parameters:
        32: Number of filters (output feature maps).
        (3, 3): Size of the filters (kernels).
        activation='relu': Introduces non-linearity. The ReLU function keeps positive values 
        unchanged and sets negative values to 0.
        input_shape=(28, 28, 1): The input is a grayscale image of size 28x28
        28x28 pixels with one channel.
        Output: Produces 32 feature maps of slightly reduced size due to the convolution.

        tf.keras.layers.MaxPooling2D((2, 2))
        Type: Max Pooling layer.
        Purpose:
        Reduces the spatial dimensions of the feature maps (downsampling).
        Helps retain the most important features (e.g., strongest edges) while reducing computational complexity.
        Parameters:
        (2, 2): Pool size. Each 2x2 2x2 region is reduced to a single maximum value.
        Output: Reduces each feature map's dimensions by half (e.g., from 28x28  to 14x14

        tf.keras.layers.Flatten()
        Type: Flattening layer.
        Purpose:
        Converts the multi-dimensional feature maps into a 1D vector.
        This step is required to feed the data into the fully connected (dense) layers, which operate on 1D vectors.
        Output: Produces a 1D vector containing all feature map values.

        tf.keras.layers.Dense(128, activation='relu')
        Type: Fully connected (dense) layer.
        Purpose:
        Performs high-level reasoning based on the extracted features.
        Learns complex patterns and relationships between the features.
        Parameters:
        128: Number of neurons in the layer.
        activation='relu': Introduces non-linearity to the dense layer.
        Output: Produces a 128-dimensional feature vector.

        tf.keras.layers.Dense(10, activation='softmax')
        Type: Output layer.
        Purpose:
        Outputs the probabilities for each of the 10 classes (e.g., digits 0-9 for MNIST digit classification).
        Parameters:
        10: Number of output neurons, corresponding to the number of classes.
        activation='softmax': Converts the raw scores (logits) into probabilities that sum to 1.
        Output: A vector of size 10, where each element represents the probability of the input belonging to a particular class.

        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  
            tf.keras.layers.MaxPooling2D((2, 2)), 
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)), 
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),  # Flatten the output to feed into Dense layers
            tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units
            tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units for 10 classes
        ])

        # Compile the model with Adam optimizer and SparseCategoricalCrossentropy loss
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        self.status(f"Retraining the model for {epochs} epochs, please wait...")
        
        # Train the model on the training data
        # x_train is an array of the images, and y_train is an array of the labels.
        self.model.fit(self.x_train, self.y_train, epochs=epochs,
                       validation_data=(self.x_test, self.y_test))

        self.status("Model retraining complete")
        self.status("Evaluating the model")
        
        # Evaluate the model on the test data to see how accurate the model is
        self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        
        # Save the retrained model to disk
        self.save_serialized_model()
        self.status(f"Pre-trained model accuracy: {self.accuracy * 100:.2f}%")
        self.status(f"Pre-trained model loss: {self.loss:.4f}")
        self.status("Model retrained and saved to disk")