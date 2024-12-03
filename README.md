# Fashion Classifier with TensorFlow UI

This project is a Fashion Image Recognition Model using TensorFlow and Keras, designed to classify images of clothing into one of 10 categories. The model is trained on the Fashion MNIST dataset and includes a user interface (UI) that allows for testing the model with test images one by one.

## Features
Convolutional Neural Network (CNN): The model is built using a CNN, which is highly effective for image classification tasks.
User Interface: A UI is provided to test the model with individual test images, displaying the predicted class for each image.
Model Persistence: The trained model is saved to disk and can be reloaded for future use, avoiding the need for retraining.

## Model Architecture

The model consists of the following layers, optimized for image classification:

Conv2D Layer:

Extracts features such as edges, corners, and textures from the input image.
Applies 32 convolutional filters of size 3x3 with ReLU activation.
Input shape: (28, 28, 1) for grayscale images.
MaxPooling2D Layer:

Reduces the spatial dimensions of the feature maps, retaining the most important features.
Pool size: 2x2.
Conv2D Layer:

Applies 64 convolutional filters of size 3x3 with ReLU activation.
MaxPooling2D Layer:

Pool size: 2x2.
Conv2D Layer:

Applies 64 convolutional filters of size 3x3 with ReLU activation.
Flatten Layer:

Converts the multi-dimensional feature maps into a 1D vector.
Dense Layer:

Fully connected layer with 128 units and ReLU activation.
Output Layer:

Fully connected layer with 10 units (one for each class) and softmax activation to output probabilities.

The layers are optimized to enhance the performance of the CNN, ensuring high accuracy in classifying fashion images.

## Usage

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/fashion-classifier-tensorflow-ui.git
    cd fashion-classifier-tensorflow-ui
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the UI**:
    ```bash
    python app.py
    ```

4. **Test Images**: Use the UI to upload and test individual images.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
