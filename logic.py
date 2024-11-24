import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np

class MNISTLogic:
    def __init__(self):
        (self.x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        self.index = 0

    def get_next_image(self):
        if self.index < len(self.x_train):
            image_array = self.x_train[self.index]
            self.index += 1
            image = Image.fromarray(image_array)
            return ImageTk.PhotoImage(image)
        return None