from Keras_Adaptive_SFM import Adaptive_SFM
from Data import make_digits_data
from keras.models import Sequential
from keras.layers import Dense

X_train, y_train, X_test, y_test = make_digits_data()

model = Sequential([
    Adaptive_SFM(X_train.shape[1],input_shape=(X_train.shape[1],X_train.shape[2])),
    Dense(y_train.shape[-1])
])

model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=128)