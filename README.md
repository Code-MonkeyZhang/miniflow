# MiniFlow Project

## Introduction

MiniFlow is a deep learning framework that mimics the style of TensorFlow, primarily implemented using the NumPy library. Users can define custom models, specifying the number of layers, learning rate, batch size, and other parameters. The framework allows users to train and perform inference with the defined models.

As part of my deep learning and AI learning journey, I hope to put the concepts learned in the course into practice. Whenever I learn a new feature or concept, such as regularization, feature scaling, etc., I can add it to this project and implement it. I believe this approach will allow me to gain a deeper understanding of these concepts and intuitively grasp the effects of those features.

## Limitations

- Currently, the framework only supports training and prediction on the MNIST dataset; support for other datasets needs to be extended
- Currently, only the basic fully connected layer and ReLU/Softmax activation functions are provided; more layer types and functionalities need to be added

## Usage Tutorial

1. Place the MNIST dataset files in the `data/mnist_data` directory, including:
   - `mnist_x_train.npy`: Training set features
   - `mnist_y_train.npy`: Training set labels
   - `mnist_x_test.npy`: Test set features
   - `mnist_y_test.npy`: Test set labels

2. Import the required libraries and modules:
```python
import os
import numpy as np
from miniflow import Model
from miniflow import Layer, FlattenLayer
```

3. Load the MNIST dataset:
```python
data_dir = os.path.join(os.path.dirname(__file__), 'data/mnist_data')
x_train = np.load(os.path.join(data_dir, 'mnist_x_train.npy'))
y_train = np.load(os.path.join(data_dir, 'mnist_y_train.npy'))
x_test = np.load(os.path.join(data_dir, 'mnist_x_test.npy'))
y_test = np.load(os.path.join(data_dir, 'mnist_y_test.npy'))
```

4. Create a model instance and define the model architecture:
```python
model = Model(
    [
        FlattenLayer(input_shape=(28, 28), layer_name='Flatten'),
        Layer(128, activation="relu", layer_name="L1", input_shape=784),
        Layer(64, activation="relu", layer_name="L2", input_shape=128),
        Layer(10, activation='softmax', layer_name="L3", input_shape=64),
    ],
    name="my_model",
    cost="softmax"
)
```

5. Train the model:
```python
model.set_rand_weight()  # Randomly initialize weights
model.fit(x_train, y_train, learning_rate=0.001, epochs=50, batch_size=32)
```

6. Evaluate the model performance on the test set:
```python
predictions = model.predict(x_test)
accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```
