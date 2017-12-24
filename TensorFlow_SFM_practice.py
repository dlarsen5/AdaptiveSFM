import tensorflow as tf
import numpy as np
from Get_Data import make_sequences


X, seq_labels = make_sequences(['AAPL'])
#X = np.asarray([[[1,2,3,4],[2,3,4,5],[2,4,31,63]],[[1,2,3,4],[2,3,4,1],[15,35,24,34]]])



def weight_variable(shape,name=None):
    if name == None:
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial,name=name)
def bias_variable(shape,name=None):
    if name==None:
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    else:
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,name=name)


Re_s_ = tf.Variable(0, name='Re_s_', dtype=tf.float32)
Im_s_ = tf.Variable(0, name='Im_s_', dtype=tf.float32)
omega_ = tf.Variable(0, name='omega_', dtype=tf.float32)

training_params = {}
features = X.shape[2]
sequence_length = X.shape[1]
z_vector_length = X.shape[1]
num_frequencies = X.shape[2]
#yooooooooooooooooooooooooooooooooo

training_params['W_state'] = weight_variable([features, sequence_length], name='W_state')
training_params['V_state'] = weight_variable([features, z_vector_length], name='V_state')
training_params['b_state'] = bias_variable([features], name='b_state')
# frequency forget weights
training_params['W_freq'] = weight_variable([num_frequencies, sequence_length], name='W_freq')
training_params['V_freq'] = weight_variable([num_frequencies, z_vector_length], name='V_freq')
training_params['b_freq'] = bias_variable([num_frequencies], name='b_freq')
# input weights
training_params['W_g'] = weight_variable([features, sequence_length], name='W_g')
training_params['V_g'] = weight_variable([features, z_vector_length], name='V_g')
training_params['b_g'] = bias_variable([features], name='b_g')
# information modulation weights
training_params['W_i'] = weight_variable([features, sequence_length], name='W_i')
training_params['V_i'] = weight_variable([features, z_vector_length], name='V_i')
training_params['b_i'] = bias_variable([features], name='b_i')
# frequency gate weights (k frequencies), omega is output for each frequency layer
training_params['U_o'] = weight_variable([num_frequencies, sequence_length, features], name='U_o')
training_params['W_o'] = weight_variable([num_frequencies, sequence_length, sequence_length], name='W_o')
training_params['V_o'] = weight_variable([num_frequencies, sequence_length, z_vector_length], name='V_o')
training_params['b_o'] = bias_variable([num_frequencies, sequence_length], name='b_o')
# final output weights
training_params['W_z'] = weight_variable([num_frequencies, sequence_length, features], name='W_z')
training_params['b_z'] = bias_variable([num_frequencies, sequence_length], name='b_z')

# for Adp
training_params['W_omega'] = weight_variable([num_frequencies, sequence_length], name='W_omega')
training_params['V_omega'] = weight_variable([num_frequencies, z_vector_length], name='V_omega')
training_params['b_omega'] = bias_variable([num_frequencies], name='b_omega')

#sequence length
x = tf.placeholder(tf.float32, shape=[z_vector_length, features])
z = tf.Variable(tf.zeros([sequence_length, features]), name='z', dtype=tf.float32)

#dimensions of D dimensional memory states
state_forget_gate = tf.sigmoid(tf.matmul(training_params['W_state'], z) + tf.matmul(training_params['V_state'], x) + training_params['b_state'], name='State_Forget')
#dimensions of K discrete frequencies
frequency_forget_gate = tf.sigmoid(tf.matmul(training_params['W_freq'], z) + tf.matmul(training_params['V_freq'], x) + training_params['b_freq'], name='Freq_Forget')

combined_forget_gate = tf.multiply(state_forget_gate, frequency_forget_gate)

input_gate = tf.sigmoid(tf.matmul(training_params['W_g'], z) + tf.matmul(training_params['V_g'], x) + training_params['b_g'],name='Input_Gate')

modulation_gate = tf.tanh(tf.matmul(training_params['W_i'], z) + tf.matmul(training_params['V_i'], x) + training_params['b_i'],name='Modulation_Gate')
# for ADP
omega = tf.matmul(training_params['W_omega'], z) + tf.matmul(training_params['V_omega'], x) + training_params['b_omega']

#both are of S(t) states wtih D dimensional memory states and K discrete frequencies
real_s = combined_forget_gate * Re_s_ + tf.multiply(input_gate * modulation_gate, tf.cos(omega_))
img_s = combined_forget_gate * Im_s_ + tf.multiply(input_gate * modulation_gate, tf.sin(omega_))

amplitude = tf.sqrt(real_s ** 2 + img_s ** 2)

# omega that combines amplitude, previous output, current input, and bias, dimensions M
def __freq():
    zz = 0
    for _ in range(num_frequencies):
        o = tf.sigmoid(
            tf.matmul(training_params['U_o'][_], amplitude) + tf.matmul(training_params['W_o'][_], z) + tf.matmul(
                training_params['V_o'][_], x) + tf.transpose(training_params['b_o']))

        zz = zz + o * tf.tanh(tf.matmul(training_params['W_z'][_], amplitude) + tf.transpose(training_params['b_z']))
    return zz

with tf.Session() as sess:
    zz = tf.Variable(0, dtype=tf.float32)
    init = tf.global_variables_initializer()

    sess.run(init)
    output = sess.run(amplitude,feed_dict={x : X[0]})
    print(output.shape)


"""
with tf.control_dependencies([o]):
    Re_s_assign = tf.assign(Re_s_,real_s)
    Im_s_assign = tf.assign(Im_s_, img_s)
    omega_assign = tf.assign(omega_, omega)
    """