## Introduction

miniflow is a deep learning framework that mimics the style of TensorFlow, primarily implemented using the NumPy library. Users can define custom models, specifying the number of layers, learning rate, batch size, and other parameters. The framework allows users to train and perform inference with the defined models.

As part of my deep learning and AI learning journey, I hope to put the concepts learned in the course into practice. Whenever I learn a new feature or concept, such as regularization, feature scaling, etc., I can add it to this project and implement it. I believe this approach will allow me to gain a deeper understanding of these concepts and intuitively grasp the effects of those features.

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

## Limitations

- Currently, the framework has only tested on the MNIST dataset; support for other datasets needs to be extended
- Currently, only the basic fully connected layer and ReLU/Softmax activation functions are provided; more layer types and functionalities need to be added
