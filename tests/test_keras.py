from Data import digits
from Keras_Adaptive_SFM import Adaptive_SFM

from keras.layers import Dense
from keras.models import Sequential


if __name__ == '__main__':

    X_train, y_train, X_test, y_test = digits()

    model = Sequential([
        Adaptive_SFM(X_train.shape[1],input_shape=(X_train.shape[1],X_train.shape[2])),
        Dense(y_train.shape[-1])
    ])

    model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=128)
