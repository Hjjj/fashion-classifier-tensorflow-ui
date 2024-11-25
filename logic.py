import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import os  # Add import for os module

TRAINING_EPOCHS = 100

class MNISTLogic:
    def __init__(self, status_label=None):  # Add status_label parameter
        self.status_label = status_label  # Store the status_label

        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        self.index = 0
        
        # Normalize the data
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        model_just_opened = False
        
        # if the model exists serialized in disk load it
        if (os.path.exists("models/fashion-ai-model.h5")):
            self.status("Model loaded from disk")
            self.model = tf.keras.models.load_model("models/fashion-ai-model.h5")
            model_just_opened = True
        else:
            # Create the model
            self.status("Creating the model")

            self.model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Softmax()  # Add softmax layer for probability distribution
            ])
            
            # Compile the model
            self.model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])
            
            # Output message to UI that training is going on
            self.status("Training the model, please wait...")

            # Train the model
            self.model.fit(self.x_train, self.y_train, epochs=TRAINING_EPOCHS)
        
        # Serialize the model
        # create a folder off of the current working directory named Models
        if not os.path.exists("models"):
            self.status("Creating models folder")
            os.makedirs("models")
        
        # save the model in the folder
        if not model_just_opened:
            self.status("Saving the model")
            self.model.save("models/fashion-ai-model.h5")

        # Evaluate the model
        self.model.evaluate(self.x_test, self.y_test, verbose=2)

    def status(self, message):
        if self.status_label:
            self.status_label.config(text=message)  # Update status label
        else:
            print(message)  # Fallback to print if no label provided

    def recognize_image(self, image_array):
        # Preprocess the image
        image_array = np.expand_dims(image_array, axis=0) / 255.0
        
        # Predict the class
        predictions = self.model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        
        return predicted_class

    def get_next_image(self):
        if self.index < len(self.x_train):
            image_array = self.x_train[self.index]
            self.index += 1
            image = Image.fromarray((image_array * 255).astype(np.uint8))  # Denormalize for display
            image_tk = ImageTk.PhotoImage(image)
            predicted_class = self.recognize_image(image_array)
            return image_tk, predicted_class
        return None, None