import tensorflow as tf
import numpy as np
#dimensions 2 x 3 x 4
X = np.asarray([[[1,2,3,4],[2,3,4,5],[2,4,31,63]],[[1,2,3,4],[2,3,4,1],[15,35,24,34]]])
Y = np.asarray([1],[2])

#x = tf.placeholder(tf.float32, shape=X.shape)
#y = tf.placeholder(tf.float32, shape=Y.shape)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class SFM():

    def __init__(self,session,observations,sequence_length,w_frequencies,input_dim):

        self._session = session
        self.training_params = {}
        # num of training/testing examples
        self.observations = observations
        # sequence length, so like features
        self.sequence_length = sequence_length
        # frequencies (K) from 1 - K
        self.w_frequencies = w_frequencies
        # total input dimensions, so m x n x k
        self.input_dim = input_dim
        self.build_graph()
        self.build_eval_graph()

    def initialize_weights(self):

        self.training_params['W_state'] = tf.Variable(tf.zeros([self.observations, self.sequence_length]))
        self.training_params['V_state'] = tf.Variable(tf.zeros([self.observations, self.input_dim]))
        self.training_params['b_state'] = tf.Variable(tf.zeros([self.observations]))
        # frequency forget weights
        self.training_params['W_freq'] = tf.Variable(tf.zeros([self.w_frequencies, self.sequence_length]))
        self.training_params['V_freq'] = tf.Variable(tf.zeros([self.w_frequencies, self.input_dim]))
        self.training_params['b_freq'] = tf.Variable(tf.zeros([self.w_frequencies]))
        # input weights
        self.training_params['W_g'] = tf.Variable(tf.zeros([self.observations, self.sequence_length]))
        self.training_params['V_g'] = tf.Variable(tf.zeros([self.observations, self.input_dim]))
        self.training_params['b_g'] = tf.Variable(tf.zeros([self.observations]))
        # information modulation weights
        self.training_params['W_i'] = tf.Variable(tf.zeros([self.observations, self.sequence_length]))
        self.training_params['V_i'] = tf.Variable(tf.zeros([self.observations, self.input_dim]))
        self.training_params['b_i'] = tf.Variable(tf.zeros([self.observations]))
        # frequency gate weights (k frequencies), omega is output for each frequency layer
        self.training_params['U_o'] = tf.Variable(tf.zeros([self.w_frequencies, self.sequence_length, self.observations]))
        self.training_params['W_o'] = tf.Variable(tf.zeros([self.w_frequencies, self.sequence_length, self.sequence_length]))
        self.training_params['V_o'] = tf.Variable(tf.zeros([self.w_frequencies, self.sequence_length, self.input_dim]))
        self.training_params['b_o'] = tf.Variable(tf.zeros([self.w_frequencies, self.sequence_length]))
        # final output weights
        self.training_params['W_z'] = tf.Variable(tf.zeros([self.w_frequencies, self.sequence_length, self.observations]))
        self.training_params['b_z'] = tf.Variable(tf.zeros([self.w_frequencies, self.sequence_length]))

        # for Adp
        self.training_params['W_omega'] = tf.Variable(tf.zeros([self.w_frequencies, self.sequence_length]))
        self.training_params['V_omega'] = tf.Variable(tf.zeros([self.w_frequencies, self.input_dim]))
        self.training_params['b_omega'] = tf.Variable(tf.zeros([self.w_frequencies]))

    def forward(self, Re_s_, Im_s_, z, x, t_, z_k, omgea_):

        state_forget_gate = tf.sigmoid(tf.matmul(self.training_params['W_state'], z) + tf.matmul(self.training_params['V_state'], x) + self.training_params['b_state'], name='State_Forget')
        frequency_forget_gate = tf.sigmoid(tf.matmul(self.training_params['W_freq'], z) + tf.matmul(self.training_params['V_freq'], x) + self.training_params['b_freq'], name='Freq_Forget')

        combined_forget_gate = tf.einsum('i,j->ij', state_forget_gate, frequency_forget_gate)

        input_gate = tf.sigmoid(tf.matmul(self.training_params['W_g'], z) + tf.matmul(self.training_params['V_g'], x) + self.training_params['b_g'],name='Input_Gate')

        modulation_gate = tf.tanh(tf.matmul(self.training_params['W_i'], z) + tf.matmul(self.training_params['V_i'], x) + self.training_params['b_i'],name='Modulation_Gate')
        # for ADP
        omega = tf.matmul(self.training_params['W_omega'], z) + tf.matmul(self.training_params['V_omega'], x) + self.training_params['b_omega']

        real_s = combined_forget_gate * Re_s_ + tf.einsum('i,j->ij', input_gate * modulation_gate, tf.cos(omgea_ * t_))
        img_s = combined_forget_gate * Im_s_ + tf.einsum('i,j->ij', input_gate * modulation_gate, tf.sin(omgea_ * t_))

        amplitude = tf.sqrt(real_s ** 2 + img_s ** 2)

        # omega that combines amplitude, previous output, current input, and bias, dimensions M
        o = tf.sigmoid(tf.matmul(self.training_params['U_o'], amplitude) + tf.matmul(self.training_params['W_o'], z) + tf.matmul(self.training_params['V_o'], x) + self.training_params['b_o'])
        # output vector for each frequency component

        zz = []
        A_k = []

        for z_, A_k in zip(z_k,A_k):
            zz.append(z_ + o * tf.tanh(tf.matmul(self.training_params['W_z'], A_k) + self.training_params['b_z']))

        zz = np.array(zz).sum()

        return zz, real_s, img_s, omega

    def loss(self):
        pass
    def optimize(self):
        pass
    def build_eval_graph(self):
        #build testing graph to predict values and compare to test values
        pass
    def build_graph(self):
        #build overall model graph




        pass
    def train(self):
        prev_log_like = -np.inf
        omega = np.array([i * 2 * np.pi / train_opt['dim_feq'] for i in range(train_opt['dim_feq'])])

        pass
    def predict(self):
        pass
    def test(self):
        pass