## Introduction

miniflow is a deep learning framework that mimics the style of TensorFlow, primarily implemented using the NumPy library. Users can define custom models, specifying the number of layers, learning rate, batch size, and other parameters. The framework allows users to train and perform inference with the defined models.


### TensorFlow Style
```python

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
```

### MiniFlow Style
```python
model = Model(
    [
        FlattenLayer(input_shape=(28, 28), layer_name='Flatten'),
        Dense(128, activation="relu", layer_name="L1", input_shape=784),
        Dense(64, activation="relu", layer_name="L2", input_shape=128),
        Dense(10, activation='softmax', layer_name="L3", input_shape=64),
    ], name="my_model", cost="softmax")

model.compile(optimizer='adam',
              alpha_decay=True,
              show_summary=False,
              plot_loss=False,
              )

model.fit(x_samples,
          y_samples,
          learning_rate=0.002,
          epochs=10,
          batch_size=32,
          b1=0.9)
```




## Quick Start

First, clone the repository to your local machine:

```bash
git clone https://github.com/Code-MonkeyZhang/miniflow.git
cd miniflow
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

Install the Local miniflow Package

```bash
pip install -e .
```

Run Tests

```bash
cd test
python minst_test.py
```
## Support List
| Dateset          | Best Test Accuracy So Far |
|---------------|-----------|
| MNIST         | 95.56%    |
| Extended MNIST | 82.82%  |

To be continued..
## Motivation
As part of my deep learning and AI learning journey, I hope to put the concepts learned in the course into practice. Whenever I learn a new feature or concept, such as regularization, feature scaling, etc., I can add it to this project and implement it. I believe this approach will allow me to gain a deeper understanding of these concepts and intuitively grasp the effects of those features.

## Limitations
- Currently, only the fully connected layer and ReLU/Softmax activation functions are provided; more layer types and functionalities are under development.
- Currently, miniflow does not support GPU acceleration, which may limit the training speed for large datasets.


