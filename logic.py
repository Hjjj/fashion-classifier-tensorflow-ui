import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np

class MNISTLogic:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        self.index = 0
        
        # Normalize the data
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        
        # Create the model
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
        
        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=5)
        
        # Evaluate the model
        self.model.evaluate(self.x_test, self.y_test, verbose=2)

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