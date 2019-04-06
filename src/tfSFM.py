import tensorflow as tf
import numpy as np


class AdaSFM():
    '''
    Adaptive State-Frequency Machine

    References
    -------
    Hu, Qi - State-Frequency Memory Recurrent Neural Networks (2017)
        - http://proceedings.mlr.press/v70/hu17c/hu17c.pdf
    '''
    def __init__(self, state_size, input_size, target_size, dtype=tf.float64):
        '''
        Build graph and expose operations as methods

        Parameters
        -------
        state_size: int
            - size of timesteps
        input_size: int
            - size of input features
        target_size: int
            - size of the output space
        dtype: tf data type
            - default tf.float64, usually follows from input data type
        '''
        self.state_size = state_size
        self.input_size = input_size
        self.target_size = target_size
        self.dtype = dtype
        self.build_graph()

    def build_graph(self):
        """
        Build TensorFlow graph and expose operations as class methods
        """
        # -------Sequence Input----
        self._inputs = tf.placeholder(self.dtype, shape=[None,
                                                         self.state_size,
                                                         self.input_size])
        self.ys = tf.placeholder(self.dtype, shape=[None,
                                                    self.target_size])
        # ----------State----------
        self.W_state = tf.Variable(tf.zeros([self.input_size,
                                             self.state_size], dtype=self.dtype))
        self.V_state = tf.Variable(tf.zeros([self.state_size,
                                             self.state_size], dtype=self.dtype))
        self.b_state = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # ---------Frequency-------
        self.W_freq = tf.Variable(tf.zeros([self.input_size,
                                            self.state_size], dtype=self.dtype))
        self.V_freq = tf.Variable(tf.zeros([self.state_size,
                                            self.state_size], dtype=self.dtype))
        self.b_freq = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # ---------Modulation------
        self.W_g = tf.Variable(tf.zeros([self.input_size,
                                         self.state_size], dtype=self.dtype))
        self.V_g = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_g = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # ---------Input-----------
        self.W_i = tf.Variable(tf.zeros([self.input_size,
                                         self.state_size], dtype=self.dtype))
        self.V_i = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_i = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # ----Persistent State-----
        self.W_omega = tf.Variable(tf.zeros([self.input_size,
                                             self.state_size], dtype=self.dtype))
        self.V_omega = tf.Variable(tf.zeros([self.state_size,
                                             self.state_size], dtype=self.dtype))
        self.b_omega = tf.Variable(tf.ones([self.state_size], dtype=self.dtype))
        # ---------Output----------
        self.U_o = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.W_o = tf.Variable(tf.zeros([self.state_size,
                                         self.input_size,
                                         self.state_size], dtype=self.dtype))
        self.V_o = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_o = tf.Variable(tf.ones([self.state_size,
                                        self.state_size], dtype=self.dtype))
        # -------Step Output-------
        self.W_z = tf.Variable(tf.zeros([self.state_size,
                                         self.state_size,
                                         self.state_size], dtype=self.dtype))
        self.b_z = tf.Variable(tf.ones([self.state_size,
                                        self.state_size], dtype=self.dtype))
        # ------Sequence Output----
        self.W_z_z = tf.Variable(tf.truncated_normal([self.state_size,
                                                      self.target_size],
                                                     dtype=self.dtype,
                                                     mean=0, stddev=.01))
        self.b_z_z = tf.Variable(tf.truncated_normal([self.target_size],
                                                     mean=1, stddev=.01,
                                                     dtype=self.dtype))

        # Init hidden state from input dimensions
        # - tf.stack() only accepts input of same dimension,
        # - Need to make 4 states of dimensions
        # - Final dim is (4, samples, timesteps, timesteps)
        self.init_hidden_t = tf.reduce_sum(self._inputs, 2)
        self.init_hidden_t = tf.expand_dims(self.init_hidden_t, axis=-1)
        self.init_hidden_t = tf.matmul(self.init_hidden_t,
                                       tf.transpose(self.init_hidden_t,
                                                    perm=[0, 2, 1]))
        self.init_hidden_t = tf.multiply(self.init_hidden_t,
                                         tf.zeros([self.state_size,
                                                   self.state_size],
                                                  dtype=self.dtype))

        self.init_hidden = tf.stack([self.init_hidden_t,
                                     self.init_hidden_t,
                                     self.init_hidden_t,
                                     self.init_hidden_t], axis=0)

        # Transpose from dimensions (samples, timesteps, inputs)
        # to dimensions (timesteps, samples, inputs)
        self._inp = tf.transpose(tf.transpose(self._inputs, perm=[2, 0, 1]))

        step_freq = tf.cast((np.arange(self.state_size, dtype=np.float64) + 1)
                            / self.state_size, dtype=self.dtype)

        out = tf.scan(self._step,
                      elems=[self._inp, step_freq],
                      initializer=self.init_hidden)

        out = tf.squeeze(out[:, -1:, :, :, -1:], axis=[1,4])
        map_fn = lambda _state: tf.nn.relu(tf.matmul(_state, self.W_z_z)
                                           + self.b_z_z)
        all_out = tf.map_fn(map_fn, out)
        last_out = all_out[-1]

        softmax = tf.nn.softmax(last_out)
        cross_entropy = -tf.reduce_mean(self.ys * tf.log(softmax))
        train_op = tf.train.AdamOptimizer().minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self.ys, 1),
                                      tf.argmax(softmax, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.loss = cross_entropy
        self.train_op = train_op
        self.accuracy = accuracy

    def _step(self, prev, input):
      '''
      One time step along the sequence time axis

      Parameters
      -------
      prev: tf.stack
      input: tf.stack

      Returns
      -------
      tf.stack
          - Need to stack output vector to match size of state for output
      '''
      x_, t_ = input
      omg_, real_, img_, z_ = tf.unstack(prev)

      omg_ = omg_[:, :, 1]
      z_ = z_[:, :, 1]

      state_fg = tf.sigmoid(tf.matmul(x_, self.W_state)
                            + tf.matmul(z_, self.V_state)
                            + self.b_state)

      freq_fg = tf.sigmoid(tf.matmul(x_,self.W_freq)
                            + tf.matmul(z_, self.V_freq)
                            + self.b_freq)

      fg = self.outer(state_fg, freq_fg)

      inp_g = tf.sigmoid(tf.matmul(x_, self.W_g)
                          + tf.matmul(z_, self.V_g)
                          + self.b_g)

      mod_g = tf.tanh(tf.matmul(x_, self.W_i)
                      + tf.matmul(z_, self.V_i)
                      + self.b_i)

      omega = tf.matmul(x_, self.W_omega) + tf.matmul(z_, self.V_omega) + self.b_omega

      real = fg * real_ + self.outer(inp_g * mod_g, tf.cos(omg_ * t_))
      img = fg * img_ + self.outer(inp_g * mod_g, tf.sin(omg_ * t_))

      amp = tf.sqrt(tf.add(tf.square(real), tf.square(img)))

      # Transpose to dim (frequency_components, samples, state) for scan
      amp = tf.transpose(amp,perm=[1,0,2])

      z = tf.scan(self.__step,
                  elems=[self.U_o, self.W_o, self.V_o,
                         self.b_o, self.W_z, self.b_z, amp],
                  initializer=tf.zeros(tf.shape(z_), dtype=self.dtype))

      z = z[-1]
      # Match dim of state matrix
      omega = tf.stack([omega for _ in range(self.state_size)], axis=1)
      z = tf.stack([z for _ in range(self.state_size)], axis=1)

      return tf.stack([omega, real, img, z])

    def __step(z_k, inputs):
      '''
      Second order step function for state

      Parameters
      -------
      z_k: tf.stack
      inputs: tf.stack
          - Dim (7, x, y, z)

      Returns
      -------
      tf.stack
      '''
      U_k, W_k, V_k, b_k, W_z_k, b_z_k, A_k = inputs

      o = tf.sigmoid(tf.matmul(A_k, U_k)
                      + tf.matmul(x_, W_k)
                      + tf.matmul(z_, V_k) + b_k)

      zz = z_k + o * tf.tanh(tf.matmul(A_k, W_z_k) + b_z_k)

      return tf.stack(zz)

    def outer(self, x, y):
      '''
      Outer Product of 2 3D matrix
      Parameters
      -------
      x: tf.mat
      y: tf.mat
      '''
      return x[:, :, np.newaxis] * y[:, np.newaxis, :]
