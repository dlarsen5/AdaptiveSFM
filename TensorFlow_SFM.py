import tensorflow as tf
import numpy as np
from numpy import newaxis

class SFM_rnn():

    def __init__(self, state_size, input_size, target_size, model_name):

        self.state_size = state_size
        self.input_size = input_size
        self.target_size = target_size
        self.model_name = model_name
        self.build_graph()

    def build_graph(self):
        """
        build the tensorflow computation graph
        expose graph operations as class methods
        """

        def outer_product(inputs):
            x, y = inputs
            outerProduct = x[:, :, newaxis] * y[:, newaxis, :]
            return outerProduct

        self._inputs = tf.placeholder(tf.float64, shape=[None, self.state_size, self.input_size])
        self.ys = tf.placeholder(tf.float64, shape=[None, self.target_size])

        self.W_state = tf.Variable(tf.zeros([self.input_size, self.state_size],dtype=tf.float64))
        self.V_state = tf.Variable(tf.zeros([self.state_size, self.state_size],dtype=tf.float64))
        self.b_state = tf.Variable(tf.ones([self.state_size],dtype=tf.float64))

        self.W_freq = tf.Variable(tf.zeros([self.input_size, self.state_size],dtype=tf.float64))
        self.V_freq = tf.Variable(tf.zeros([self.state_size, self.state_size],dtype=tf.float64))
        self.b_freq = tf.Variable(tf.ones([self.state_size],dtype=tf.float64))

        self.W_g = tf.Variable(tf.zeros([self.input_size, self.state_size],dtype=tf.float64))
        self.V_g = tf.Variable(tf.zeros([self.state_size, self.state_size],dtype=tf.float64))
        self.b_g = tf.Variable(tf.ones([self.state_size],dtype=tf.float64))

        self.W_i = tf.Variable(tf.zeros([self.input_size, self.state_size],dtype=tf.float64))
        self.V_i = tf.Variable(tf.zeros([self.state_size, self.state_size],dtype=tf.float64))
        self.b_i = tf.Variable(tf.ones([self.state_size],dtype=tf.float64))

        self.W_omega = tf.Variable(tf.zeros([self.input_size, self.state_size], dtype=tf.float64))
        self.V_omega = tf.Variable(tf.zeros([self.state_size, self.state_size], dtype=tf.float64))
        self.b_omega = tf.Variable(tf.ones([self.state_size], dtype=tf.float64))

        self.U_o = tf.Variable(tf.zeros([self.state_size, self.state_size, self.state_size],dtype=tf.float64))
        self.W_o = tf.Variable(tf.zeros([self.state_size, self.input_size, self.state_size],dtype=tf.float64))
        self.V_o = tf.Variable(tf.zeros([self.state_size, self.state_size, self.state_size],dtype=tf.float64))
        self.b_o = tf.Variable(tf.ones([self.state_size, self.state_size],dtype=tf.float64))

        self.W_z = tf.Variable(tf.zeros([self.state_size, self.state_size, self.state_size],dtype=tf.float64))
        self.b_z = tf.Variable(tf.ones([self.state_size, self.state_size],dtype=tf.float64))

        #final output, map from state_size to target_size
        self.W_z_z = tf.Variable(tf.truncated_normal([self.state_size, self.target_size],dtype=tf.float64, mean=0, stddev=.01))
        self.b_z_z = tf.Variable(tf.truncated_normal([self.target_size], mean=1, stddev=.01, dtype=tf.float64))

        def step(prev_output, input):

            x_, t_ = input
            #get previous state variables
            omg_, Re_s_, Im_s_, z_ = tf.unstack(prev_output)
            #Only need one vector from matrix for forward step
            omg_ = omg_[:, :, 1]
            z_ = z_[:, :, 1]

            state_forget_gate = tf.sigmoid(tf.matmul(x_, self.W_state) + tf.matmul(z_, self.V_state) + self.b_state)
            frequency_forget_gate = tf.sigmoid(tf.matmul(x_,self.W_freq) + tf.matmul(z_, self.V_freq) + self.b_freq)

            combined_forget_gate = outer_product([state_forget_gate, frequency_forget_gate])

            input_gate = tf.sigmoid(tf.matmul(x_, self.W_g) + tf.matmul(z_, self.V_g) + self.b_g)

            modulation_gate = tf.tanh(tf.matmul(x_, self.W_i) + tf.matmul(z_, self.V_i) + self.b_i)

            omega = tf.matmul(x_, self.W_omega) + tf.matmul(z_, self.V_omega) + self.b_omega

            real_s = combined_forget_gate * Re_s_ + outer_product([input_gate * modulation_gate, tf.cos(omg_ * t_)])
            img_s = combined_forget_gate * Im_s_ + outer_product([input_gate * modulation_gate, tf.sin(omg_ * t_)])

            amplitude = tf.sqrt(tf.add(tf.square(real_s),tf.square(img_s)))

            def __freq(z_k,inputs):

                U_k, W_k, V_k, b_k, W_z_k, b_z_k, A_k = inputs
                o = tf.sigmoid(tf.matmul(A_k, U_k) + tf.matmul(x_, W_k) + tf.matmul(z_, V_k) + b_k)
                zz = z_k + o * tf.tanh(tf.matmul(A_k, W_z_k) + b_z_k)

                return tf.stack(zz)

            amplitude = tf.transpose(amplitude,perm=[1,0,2])

            new_z = tf.scan(__freq,elems=[self.U_o,self.W_o,self.V_o,self.b_o,self.W_z,self.b_z,amplitude],initializer=tf.zeros(tf.shape(z_),dtype=tf.float64))
            #Get last output of iterating through summation of frequency components
            new_z = new_z[-1]

            omega = tf.stack([omega for _ in range(self.state_size)],axis=1)
            new_z = tf.stack([new_z for _ in range(self.state_size)],axis=1)

            return tf.stack([omega, real_s, img_s, new_z])

        def get_output(hidden_state):
            return tf.nn.relu(tf.matmul(hidden_state,self.W_z_z) + self.b_z_z)

        #generate initial hidden state, since tf.stack() only accepts same dimensional input, need to make 4 states of dimension (samples, timesteps, timesteps)
        self.initial_hidden_time = tf.reduce_sum(self._inputs,2)
        self.initial_hidden_time = tf.expand_dims(self.initial_hidden_time,axis=-1)
        self.initial_hidden_time = tf.matmul(self.initial_hidden_time,tf.transpose(self.initial_hidden_time,perm=[0,2,1]))
        self.initial_hidden_time = tf.multiply(self.initial_hidden_time,tf.zeros([self.state_size, self.state_size],dtype=tf.float64))
        self.initial_hidden_time = self.initial_hidden_time + 1

        self.initial_hidden = tf.stack([self.initial_hidden_time, self.initial_hidden_time, self.initial_hidden_time, self.initial_hidden_time],axis=0)
        #transpose from dimensions (timesteps, samples, inputs) to dimensions (timesteps, samples, inputs)
        self.processed_input = tf.transpose(tf.transpose(self._inputs,perm=[2,0,1]))
        outputs = tf.scan(step, elems=[self.processed_input,(np.arange(self.state_size, dtype=np.float64) + 1) / self.state_size], initializer=self.initial_hidden)

        outputs = tf.squeeze(outputs[:, -1:, :, :, -1:],axis=[1,4])
        all_outputs = tf.map_fn(get_output,outputs)
        last_output = all_outputs[-1]

        final_output = tf.nn.softmax(last_output)
        #crossentropy loss function
        cross = -tf.reduce_mean(self.ys * tf.log(final_output))
        #Adam optimizer
        train_op = tf.train.AdamOptimizer().minimize(cross)

        correct_prediction = tf.equal(tf.argmax(self.ys, 1), tf.argmax(final_output, 1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

        self.loss = cross
        self.train_op = train_op
        self.accuracy = accuracy