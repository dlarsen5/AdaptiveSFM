import tensorflow as tf
import numpy as np
from numpy import newaxis

class Adaptive_SFM():

    def __init__(self, state_size, input_size, target_size, dtype=tf.float64):
        """Adaptive State-Frequency Memory Recurrent Neural Network - Hu, Qi 2017.

        # Arguments
            state_size: Positive integer, number of timesteps.
            input_size: Positive integer, number of input features.
            target_size: Positive integer, dimensionality of the output space.
            dtype: data type (default tf.float64. Usually follows from input data type.

        # References
        - [State-Frequency Memory Recurrent Neural Networks] (http://proceedings.mlr.press/v70/hu17c/hu17c.pdf)
        """

        self.state_size = state_size
        self.input_size = input_size
        self.target_size = target_size
        self.dtype = dtype
        self.build_graph()

    def build_graph(self):
        """
        build the tensorflow computation graph
        and expose graph operations as class methods
        """

        def outer_product(x,y):
            return x[:, :, newaxis] * y[:, newaxis, :]

        self._inputs = tf.placeholder(self.dtype, shape=[None, self.state_size, self.input_size])
        self.ys = tf.placeholder(self.dtype, shape=[None, self.target_size])

        self.W_state = tf.Variable(tf.zeros([self.input_size, self.state_size],dtype=self.dtype))
        self.V_state = tf.Variable(tf.zeros([self.state_size, self.state_size],dtype=self.dtype))
        self.b_state = tf.Variable(tf.ones([self.state_size],dtype=self.dtype))

        self.W_freq = tf.Variable(tf.zeros([self.input_size, self.state_size],dtype=self.dtype))
        self.V_freq = tf.Variable(tf.zeros([self.state_size, self.state_size],dtype=self.dtype))
        self.b_freq = tf.Variable(tf.ones([self.state_size],dtype=self.dtype))

        self.W_g = tf.Variable(tf.zeros([self.input_size, self.state_size],dtype=self.dtype))
        self.V_g = tf.Variable(tf.zeros([self.state_size, self.state_size],dtype=self.dtype))
        self.b_g = tf.Variable(tf.ones([self.state_size],dtype=self.dtype))

        self.W_i = tf.Variable(tf.zeros([self.input_size, self.state_size],dtype=self.dtype))
        self.V_i = tf.Variable(tf.zeros([self.state_size, self.state_size],dtype=self.dtype))
        self.b_i = tf.Variable(tf.ones([self.state_size],dtype=self.dtype))

        self.W_omega = tf.Variable(tf.zeros([self.input_size, self.state_size], dtype=self.dtype))
        self.V_omega = tf.Variable(tf.zeros([self.state_size, self.state_size], dtype=self.dtype))
        self.b_omega = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))

        self.U_o = tf.Variable(tf.zeros([self.state_size, self.state_size, self.state_size],dtype=self.dtype))
        self.W_o = tf.Variable(tf.zeros([self.state_size, self.input_size, self.state_size],dtype=self.dtype))
        self.V_o = tf.Variable(tf.zeros([self.state_size, self.state_size, self.state_size],dtype=self.dtype))
        self.b_o = tf.Variable(tf.ones([self.state_size, self.state_size],dtype=self.dtype))

        self.W_z = tf.Variable(tf.zeros([self.state_size, self.state_size, self.state_size],dtype=self.dtype))
        self.b_z = tf.Variable(tf.ones([self.state_size, self.state_size],dtype=self.dtype))

        # final output, map from state_size to target_size
        self.W_z_z = tf.Variable(tf.truncated_normal([self.state_size, self.target_size],dtype=self.dtype, mean=0, stddev=.01))
        self.b_z_z = tf.Variable(tf.truncated_normal([self.target_size], mean=1, stddev=.01, dtype=self.dtype))

        def step(prev_output, input):

            x_, t_ = input

            omg_, Re_s_, Im_s_, z_ = tf.unstack(prev_output)
            # only need one vector from matrix for forward step
            omg_ = omg_[:, :, 1]
            z_ = z_[:, :, 1]

            state_forget_gate = tf.sigmoid(tf.matmul(x_, self.W_state) + tf.matmul(z_, self.V_state) + self.b_state)
            frequency_forget_gate = tf.sigmoid(tf.matmul(x_,self.W_freq) + tf.matmul(z_, self.V_freq) + self.b_freq)

            combined_forget_gate = outer_product(state_forget_gate, frequency_forget_gate)

            input_gate = tf.sigmoid(tf.matmul(x_, self.W_g) + tf.matmul(z_, self.V_g) + self.b_g)

            modulation_gate = tf.tanh(tf.matmul(x_, self.W_i) + tf.matmul(z_, self.V_i) + self.b_i)

            omega = tf.matmul(x_, self.W_omega) + tf.matmul(z_, self.V_omega) + self.b_omega

            real_s = combined_forget_gate * Re_s_ + outer_product(input_gate * modulation_gate, tf.cos(omg_ * t_))
            img_s = combined_forget_gate * Im_s_ + outer_product(input_gate * modulation_gate, tf.sin(omg_ * t_))

            amplitude = tf.sqrt(tf.add(tf.square(real_s),tf.square(img_s)))

            def __freq(z_k,inputs):

                U_k, W_k, V_k, b_k, W_z_k, b_z_k, A_k = inputs
                o = tf.sigmoid(tf.matmul(A_k, U_k) + tf.matmul(x_, W_k) + tf.matmul(z_, V_k) + b_k)
                zz = z_k + o * tf.tanh(tf.matmul(A_k, W_z_k) + b_z_k)

                return tf.stack(zz)
            # transpose to dimensions (frequency_components, samples, state) for tf.scan
            amplitude = tf.transpose(amplitude,perm=[1,0,2])

            new_z = tf.scan(__freq,elems=[self.U_o,self.W_o,self.V_o,self.b_o,self.W_z,self.b_z,amplitude],initializer=tf.zeros(tf.shape(z_),dtype=self.dtype))
            # get last output of iterating through summation of frequency components
            new_z = new_z[-1]
            # make new omega and h matrices to fit size of other stacked matrices
            omega = tf.stack([omega for _ in range(self.state_size)],axis=1)
            new_z = tf.stack([new_z for _ in range(self.state_size)],axis=1)

            return tf.stack([omega, real_s, img_s, new_z])

        # generate initial hidden state from unknown input dimensions
        # since tf.stack() only accepts input of same dimension,
        # need to make 4 states of dimension (samples, timesteps, timesteps)
        self.initial_hidden_time = tf.reduce_sum(self._inputs,2)
        self.initial_hidden_time = tf.expand_dims(self.initial_hidden_time,axis=-1)
        self.initial_hidden_time = tf.matmul(self.initial_hidden_time,tf.transpose(self.initial_hidden_time,perm=[0,2,1]))
        self.initial_hidden_time = tf.multiply(self.initial_hidden_time,tf.zeros([self.state_size, self.state_size],dtype=self.dtype))

        self.initial_hidden = tf.stack([self.initial_hidden_time, self.initial_hidden_time, self.initial_hidden_time, self.initial_hidden_time],axis=0)
        # transpose from dimensions (samples, timesteps, inputs) to dimensions (timesteps, samples, inputs)
        self.processed_input = tf.transpose(tf.transpose(self._inputs,perm=[2,0,1]))
        outputs = tf.scan(step, elems=[self.processed_input,tf.cast((np.arange(self.state_size, dtype=np.float64) + 1) / self.state_size,dtype=self.dtype)], initializer=self.initial_hidden)

        outputs = tf.squeeze(outputs[:, -1:, :, :, -1:],axis=[1,4])
        all_outputs = tf.map_fn(lambda hidden_state: tf.nn.relu(tf.matmul(hidden_state, self.W_z_z) + self.b_z_z), outputs)
        last_output = all_outputs[-1]

        final_output = tf.nn.softmax(last_output)
        # loss
        cross_entropy = -tf.reduce_mean(self.ys * tf.log(final_output))
        # optimize op
        train_op = tf.train.AdamOptimizer().minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self.ys, 1), tf.argmax(final_output, 1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

        self.loss = cross_entropy
        self.train_op = train_op
        self.accuracy = accuracy