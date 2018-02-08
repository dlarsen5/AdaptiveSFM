# State-Frequency Memory Recurrent Neural Network in TensorFlow and Keras

From[Hu, Qi (2017)](http://proceedings.mlr.press/v70/hu17c/hu17c.pdf), a state-frequency memory recurrent 
neural network implemented in TensorFlow and Keras. Implementation in Keras makes it easy to add additional recurrent
layers of any type and experimentation with different loss functions and optimizers.

## Getting Started

Make sure you have the dependencies installed and the './datasets' folder in the same root directory. Can use the Keras
and TensorFlow scripts independently or run the example scripts to see the network in action. The examples gather digits
data from sklearn's[digits data](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits).
You can also generate text sequence data for training by utilizing [Data.make_text_data()] but due to the nature of the 
network, this can significantly increase computation time.

Also make sure you check out the original paper, very neat stuff.

## Built With

* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [Numpy](http://www.numpy.org/)
* [Scikit-Learn](http://scikit-learn.org/stable/)