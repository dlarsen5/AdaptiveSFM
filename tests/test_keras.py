import sys
sys.path.append('src/')

from Data import digits
from kerasSFM import AdaSFM

from keras.layers import Dense
from keras.models import Sequential

EPOCHS = 100
BATCH = 128


if __name__ == '__main__':

    X_train, y_train, X_test, y_test = digits()

    sfm = Sequential([Adaptive_SFM(X_train.shape[1],
                                   input_shape=(X_train.shape[1],
                                                X_train.shape[2])),
                      Dense(y_train.shape[-1])])

    sfm.compile(loss='mean_squared_error',
                optimizer='Adam',
                metrics=['accuracy'])

    sfm.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH)
