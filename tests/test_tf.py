import sys
sys.path.append('src/')

from Data import digits
from tfSFM import AdaSFM

import tensorflow as tf


EPOCHS = 100
BATCH = 128


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = digits()

    sfm = AdaSFM(state_size=x_train.shape[1],
                     input_size=x_train.shape[2],
                     target_size=y_train.shape[1])

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            loss = 0
            mini_start = 0
            mini_end = BATCH

            for i in range(0, x_train.shape[0], BATCH):
                batch_X = x_train[mini_start:mini_end]
                batch_Y = y_train[mini_start:mini_end]

                mini_start = mini_end
                mini_end = mini_start + BATCH

                loss_, _ = sess.run([sfm.loss, sfm.train_op],
                                    feed_dict={sfm._inputs: batch_X,
                                               sfm.ys: batch_Y})

                loss += loss_


            test_accuracy = sess.run(sfm.accuracy,
                                     feed_dict={sfm._inputs: x_test,
                                                sfm.ys: y_test})
            test_loss = loss / (x_train.shape[0] / BATCH)

            print("Epoch: %s Training Loss: %s Test Accuracy: %s" % (epoch,
                                                                     test_loss,
                                                                     test_accuracy))
