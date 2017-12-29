import tensorflow as tf
import numpy as np
from Get_Data import make_sequences

class SFM_rnn():

    def __init__(self,dimension_D,dimension_N, model_name):

        self.dimension_D = dimension_D
        self.dimension_H = dimension_D
        self.dimension_N = dimension_N
        self.model_name = model_name
        self.build_graph()

    def build_graph(self):

        x = tf.placeholder(tf.float32, shape=[None, self.dimension_D, self.dimension_N])
        y = tf.placeholder(tf.float32, shape=[None, self.dimension_D])

        with tf.variable_scope('Cell', reuse=False):
            W_state = tf.get_variable('W_state', shape=[self.dimension_D, self.dimension_N])
            V_state = tf.get_variable('V_state', shape=[self.dimension_H, self.dimension_H])
            b_state = tf.get_variable('b_state', shape=[self.dimension_H])

            W_freq = tf.get_variable('W_freq', shape=[self.dimension_D, self.dimension_N])
            V_freq = tf.get_variable('V_freq', shape=[self.dimension_H, self.dimension_H])
            b_freq = tf.get_variable('b_freq', shape=[self.dimension_H])

            W_g = tf.get_variable('W_g', shape=[self.dimension_D, self.dimension_N])
            V_g = tf.get_variable('V_g', shape=[self.dimension_H, self.dimension_H])
            b_g = tf.get_variable('b_g', shape=[self.dimension_H])

            W_i = tf.get_variable('W_i', shape=[self.dimension_D, self.dimension_N])
            V_i = tf.get_variable('V_i', shape=[self.dimension_H, self.dimension_H])
            b_i = tf.get_variable('b_i', shape=[self.dimension_H])

            U_o = tf.get_variable('U_o', shape=[self.dimension_H, self.dimension_H, self.dimension_H])
            W_o = tf.get_variable('W_o', shape=[self.dimension_H, self.dimension_D, self.dimension_N])
            V_o = tf.get_variable('V_o', shape=[self.dimension_H, self.dimension_H, self.dimension_H])
            b_o = tf.get_variable('b_o', shape=[self.dimension_H, self.dimension_H])


            W_z = tf.get_variable('W_z', shape=[self.dimension_H, self.dimension_H, self.dimension_H])
            b_z = tf.get_variable('b_z', shape=[self.dimension_H, self.dimension_H])

            W_omega = tf.get_variable('W_omega', shape=[self.dimension_D, self.dimension_N])
            V_omega = tf.get_variable('V_omega', shape=[self.dimension_H, self.dimension_H])
            b_omega = tf.get_variable('b_omega', shape=[self.dimension_H])

        def step(prev_output, x_):

            omg_, Re_s_, Im_s_, z_ = tf.unstack(prev_output)

            state_forget_gate = tf.sigmoid(tf.reduce_sum(tf.multiply(W_state, x_), 1, keep_dims=True) + tf.multiply(V_state, z_) + b_state, name='State_Forget')
            frequency_forget_gate = tf.sigmoid(tf.reduce_sum(tf.multiply(W_freq, x_), 1, keep_dims=True) + tf.multiply(V_freq, z_) + b_freq, name='Freq_Forget')

            combined_forget_gate = tf.multiply(state_forget_gate, frequency_forget_gate)

            input_gate = tf.sigmoid(tf.reduce_sum(tf.multiply(W_g, x_), 1, keep_dims=True) + tf.multiply(V_g, z_) + b_g, name='Input_Gate')

            modulation_gate = tf.tanh(tf.reduce_sum(tf.multiply(W_i, x_), 1, keep_dims=True) + tf.multiply(V_i, z_) + b_i, name='Modulation_Gate')

            omega = tf.reduce_sum(tf.multiply(W_omega, x_), 1, keep_dims=True) + tf.multiply(V_omega, z_) + b_omega

            real_s = combined_forget_gate * Re_s_ + tf.multiply(input_gate * modulation_gate, tf.cos(omg_)) #todo: figure out t thing
            img_s = combined_forget_gate * Im_s_ + tf.multiply(input_gate * modulation_gate, tf.sin(omg_))


            amplitude = tf.sqrt(real_s ** 2 + img_s ** 2)

            def __freq(prev,inputs):

                U_k, W_k, V_k, b_k, W_z_k, b_z_k, A_k = inputs
                z_k = tf.unstack(prev)
                o = tf.sigmoid(tf.multiply(U_k, A_k) + tf.reduce_sum(tf.multiply(W_k, x_)) + tf.multiply(V_k, z_) + b_k)

                zz = z_k + o * tf.tanh(tf.multiply(W_z_k, A_k) + b_z_k)

                return tf.stack(zz)

            initial_z = np.ones((X.shape[1],X.shape[1]), dtype=np.float32)
            new_z = tf.scan(__freq,elems=[U_o,W_o,V_o,b_o,W_z,b_z,tf.transpose(amplitude)],initializer=initial_z)
            new_z = tf.reduce_mean(new_z,0)

            return tf.stack([omega, real_s, img_s, new_z])

        initial_weights = np.ones((4, self.dimension_D, self.dimension_D), dtype=np.float32)

        outputs = tf.scan(step, elems=x, initializer=initial_weights)

        states = outputs[:, -1:, :, :]

        states = tf.squeeze(states, axis=1)
        logits = tf.reduce_sum(states, axis=1)
        predictions = tf.nn.softmax(logits)
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        #todo figure out logits
        #losses = tf.Print(losses,data=[predictions])
        loss = tf.reduce_mean(losses)
        #todo loss is only increasing, fix learning
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

        self.xs = x
        self.ys = y
        self.states = states
        self.predictions = predictions
        self.loss = loss
        self.train_op = train_op

    def train(self, train_set, epochs=100, batch_size=128):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            try:
                for i in range(epochs):
                    xs, ys = train_set
                    for offset in range(0, len(xs), batch_size):
                        batch_x = xs[offset: offset + batch_size]
                        batch_y = ys[offset: offset + batch_size]
                        #train_loss_ = sess.run(self.loss,
                         #                         feed_dict={self.xs: batch_x, self.ys: batch_y})
                        _, train_loss_ = sess.run([self.train_op,self.loss], feed_dict={self.xs : batch_x, self.ys : batch_y})
                        train_loss += train_loss_
                    print('[{}] loss: {}'.format(i,train_loss/100))
                    train_loss = 0
            except KeyboardInterrupt:
                print('Interrupted by user')

#X, seq_labels = make_sequences(['AAPL'])

X = np.random.rand(5, 5, 3).astype(np.float32)
seq_labels = np.random.rand(5, 5).astype(np.float32)

X = X.astype(np.float32)

SFM = SFM_rnn(dimension_D=X.shape[1],dimension_N=X.shape[2], model_name='SFM_1')
SFM.train(train_set=[X,seq_labels],epochs=10)