import os

from keras.utils import np_utils
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def digits(test_size=0.22):
    """
    Make digits train/test sequences from sklearn.datasets

    Parameters
    -------
    test_size: float
        - test/train split percentage

    Returns
    -------
    X_train, y_train, X_test, y_test: np.Array
    """
    def get_one_hot(number, digits=10):
        one_hot = [0] * digits
        one_hot[number] = 1

        return one_hot

    digits = datasets.load_digits()
    X = digits.images
    Y_ = digits.target

    Y = [get_one_hot(x) for x in Y_]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    X_train = np.array(X_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    return X_train, y_train, X_test, y_test

def text(test_size=0.22, file_index=0):
    """
    Make text train/test sequences from text files in folder '/datasets'

    Parameters
    -------
    test_size: float
        - test/train split percentage
    data_set: int
        - index position of text file in os.listdir('../datasets')

    Returns
    -------
    X_train, y_train, X_test, y_test: np.Array
        - Sequential text data in X/Y train/test arrays
    """
    data_path = os.path.curdir + '/datasets/'
    files = os.listdir(data_path)

    if file_index in range(len(files)):
        filename = data_path + files[file_index]
        raw_text = open(filename).read()
        raw_text = raw_text.lower()
    else:
        raise ValueError("file_index must be in range of index of os.listdir('../datasets')")

    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    n_chars = len(raw_text)
    n_vocab = len(chars)

    seq_length = 30
    dataX = []
    dataY = []

    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    test_x = X
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    Y = np_utils.to_categorical(dataY)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    X_train = np.array(X_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    return X_train, y_train, X_test, y_test
