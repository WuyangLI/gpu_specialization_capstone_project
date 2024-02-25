# Linear Algebra MNIST Classifier using cuBlas

## Project Description

A simple MNIST classifier using cuBlas
the classifier consists of two MLP layers and relu activation
it reads the grey-sclae image of MNIST image, normalizes image tensor to [0, 1], pass the tensor (float array in implementation) through the two layer MLP, and then calculate the softmax classification score. The index of which the respective element value is the greatest is the prediction of the classifier.


in algebra, the model prediction P is a function of input X and weights W1, W2, depicted as below:
        P = softmax(relu(X * W1) * W2)

After 10 epochs of training, training loss is around: 1.6
accuracy of the model is around: 64.35 %

Please note that the accuracy of random-guess is 10%.

## Code Organization

```data/```
This folder holds traning and testing MNIST data in binary format. You can use the python script `generate_train_test_data.py` to generate readable txt dataset.

```src/```
There are two files in src. 
- the prototype of the classifier
  This is implemented in numpy for verifying the correctness of the algorithm, especially the `backPropagation`. As we implement the classifier from scratch, there is no `Automatic differentiation` system readily avilable to use. Thus we manually derieve the derivative of the matrixes and build backpropagation machenism for this simple classifier.

- The MNIST classifier using cuBlas
  this file is basically cuda version of the python prototype.
  element-wise operations are implemented from scratch as cuBlas doesn't support them.
  matrix linear algebra operations are implemented using cuBlas api.
  
```README.md```
This file holds the description of the project

```Makefile```
you can use `make build` to build the code

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.
