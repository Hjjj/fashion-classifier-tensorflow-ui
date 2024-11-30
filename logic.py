import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import os  # Add import for os module

TRAINING_EPOCHS = 5
MODEL_PATH = "models/fashion-ai-model.keras"  # Add constant for model path
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Add class labels

class AIFashionImageRecognitionModel:
    def __init__(self, status_label=None):  # Add status_label parameter
        self.status_label = status_label  # Store the status_label
        self.index = 0 #this var keeps state of the current image being processed
        self.load_fashion_data()  # fashion MNIST images with labels
        
     
        if not self.open_serialized_model():
            self.status("Model not found, retraining...")
            self.retrain(TRAINING_EPOCHS)
        
        self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print(f"Pre-trained model accuracy: {self.accuracy * 100:.2f}%")
        print(f"Pre-trained model loss: {self.loss:.4f}")

        self.save_serialized_model()

    def load_fashion_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        #self.x_train = np.expand_dims(self.x_train, -1)
        #self.x_test = np.expand_dims(self.x_test, -1)
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28, 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28, 1))



    def open_serialized_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.status(f"Model loading from {MODEL_PATH}")
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.model.compile(optimizer='adam',  # Compile the model after loading
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                   metrics=['accuracy'])
                return True
            except OSError as e:
                self.status(f"Error loading model: {e}")
                self.model = None
        return False

    def save_serialized_model(self):
        # create a folder off of the current working directory named Models
        if not os.path.exists("models"):
            self.status("Creating models folder")
            os.makedirs("models")
        
        # save the model in the folder
        self.status("Saving the model")
        self.model.save(MODEL_PATH)

    def status(self, message):
        if self.status_label:
            self.status_label.config(text=message)  # Update status label
        print(message)

    def recognize_image(self, image_array):
        # Preprocess the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Predict the class
        predictions = self.model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]  # Convert to string label
        
        return predicted_label

    def interpret_next_image(self):
        if self.index < len(self.x_test):
            image_array = self.x_test[self.index]
            self.index += 1
            image = Image.fromarray((np.squeeze(image_array) * 255).astype(np.uint8))  # Denormalize for display
            image_tk = ImageTk.PhotoImage(image)
            predicted_label = self.recognize_image(image_array)
            return image_tk, predicted_label
        return None, None

    def retrain(self, epochs):
        # Create the CNN model
        self.status("Creating the CNN model")

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            #tf.keras.layers.Dropout(0.5), #remove?? TODO
            tf.keras.layers.Dense(10, activation='softmax')  # Add softmax layer for probability distribution
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        self.status(f"Retraining the model for {epochs} epochs, please wait...")
        self.model.fit(self.x_train, self.y_train, epochs=epochs,
                       validation_data=(self.x_test, self.y_test))

        self.status("Model retraining complete")
        self.status("Evaluating the model")
        
        self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        self.save_serialized_model()
        self.status(f"Pre-trained model accuracy: {self.accuracy * 100:.2f}%")
        self.status(f"Pre-trained model loss: {self.loss:.4f}")
        self.status("Model retrained and saved to disk")