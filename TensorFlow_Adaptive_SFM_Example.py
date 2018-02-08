from TensorFlow_Adaptive_SFM import Adaptive_SFM
from Data import make_digits_data
import tensorflow as tf

epochs = 100
batch_size = 128

X_train, y_train, X_test, y_test = make_digits_data()

Ada_SFM = Adaptive_SFM(state_size=X_train.shape[1],input_size=X_train.shape[2], target_size=y_train.shape[1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        loss = 0
        mini_start = 0
        mini_end = batch_size
        # mini batches
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[mini_start:mini_end]
            batch_Y = y_train[mini_start:mini_end]
            mini_start = mini_end
            mini_end = mini_start + batch_size
            loss_, _ = sess.run([Ada_SFM.loss, Ada_SFM.train_op], feed_dict={Ada_SFM._inputs: batch_X, Ada_SFM.ys: batch_Y})
            loss += loss_
        test_accuracy = sess.run(Ada_SFM.accuracy, feed_dict={Ada_SFM._inputs: X_test, Ada_SFM.ys: y_test})
        print("Epoch: %s Training Loss: %s Test Accuracy: %s" % (epoch, loss / (X_train.shape[0] / batch_size), test_accuracy))