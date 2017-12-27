import tensorflow as tf
import numpy as np
from Get_Data import make_sequences

def build_graph():
    with tf.variable_scope('Cell'):

        W_pre_state = tf.get_variable('W_pre_state', shape=[dimension_D, dimension_N])
        W_state = tf.get_variable('W_state', shape=[dimension_D])
        V_state = tf.get_variable('V_state', shape=[dimension_H])
        b_state = tf.get_variable('b_state', shape=[dimension_H])

        W_pre_freq = tf.get_variable('W_pre_freq', shape=[dimension_D, dimension_N])
        W_freq = tf.get_variable('W_freq', shape=[dimension_D])
        V_freq = tf.get_variable('V_freq', shape=[dimension_H])
        b_freq = tf.get_variable('b_freq', shape=[dimension_H])

        W_pre_g = tf.get_variable('W_pre_g', shape=[dimension_D, dimension_N])
        W_g = tf.get_variable('W_g', shape=[dimension_D])
        V_g = tf.get_variable('V_g', shape=[dimension_H])
        b_g = tf.get_variable('b_g', shape=[dimension_H])


        W_pre_i = tf.get_variable('V_pre_i', shape=[dimension_D, dimension_N])
        W_i = tf.get_variable('W_i', shape=[dimension_D])
        V_i = tf.get_variable('V_i', shape=[dimension_H])
        b_i = tf.get_variable('b_i', shape=[dimension_H])

        #U_o = tf.get_variable('U_o', shape=[dimension_H, dimension_H, dimension_H])
        #W_pre_o = tf.get_variable('W_pre_o', shape=[dimension_D, dimension_N])
        #W_o = tf.get_variable('W_o', shape=[dimension_H, dimension_H, dimension_D])
        #V_o = tf.get_variable('V_o', shape=[dimension_H, dimension_H, dimension_H])
        #b_o = tf.get_variable('b_i', shape=[dimension_H, dimension_H])


        #W_z = tf.get_variable('W_z', shape=[dimension_H, dimension_H, dimension_H]nsion_H], stddev=0.1))
        #b_z = tf.get_variable('b_z', shape=[dimension_H, dimension_H]imension_H]))

        W_pre_omega = tf.get_variable('W_pre_omega', shape=[dimension_D, dimension_N])
        W_omega = tf.get_variable('W_omega', shape=[dimension_H, dimension_D])
        V_omega = tf.get_variable('V_omega', shape=[dimension_H, dimension_H])
        b_omega = tf.get_variable('b_omega', shape=[dimension_H])

    return

def step(prev_output, x):
    #TODO sort out prev value unpacking
    omg_, Re_s_, Im_s_, amp = tf.unstack(prev_output)

    with tf.variable_scope('Cell',reuse=True):

        W_pre_state = tf.get_variable('W_pre_state', shape=[dimension_D, dimension_N])
        W_state = tf.get_variable('W_state', shape=[dimension_D])
        V_state = tf.get_variable('V_state', shape=[dimension_H])
        b_state = tf.get_variable('b_state', shape=[dimension_H])

        W_pre_freq = tf.get_variable('W_pre_freq', shape=[dimension_D, dimension_N])
        W_freq = tf.get_variable('W_freq', shape=[dimension_D])
        V_freq = tf.get_variable('V_freq', shape=[dimension_H])
        b_freq = tf.get_variable('b_freq', shape=[dimension_H])

        W_pre_g = tf.get_variable('W_pre_g', shape=[dimension_D, dimension_N])
        W_g = tf.get_variable('W_g', shape=[dimension_D])
        V_g = tf.get_variable('V_g', shape=[dimension_H])
        b_g = tf.get_variable('b_g', shape=[dimension_H])


        W_pre_i = tf.get_variable('V_pre_i', shape=[dimension_D, dimension_N])
        W_i = tf.get_variable('W_i', shape=[dimension_D])
        V_i = tf.get_variable('V_i', shape=[dimension_H])
        b_i = tf.get_variable('b_i', shape=[dimension_H])

        #U_o = tf.get_variable('U_o', shape=[dimension_H, dimension_H, dimension_H])
        #W_pre_o = tf.get_variable('W_pre_o', shape=[dimension_D, dimension_N])
        #W_o = tf.get_variable('W_o', shape=[dimension_H, dimension_H, dimension_D])
        #V_o = tf.get_variable('V_o', shape=[dimension_H, dimension_H, dimension_H])
        #b_o = tf.get_variable('b_i', shape=[dimension_H, dimension_H])


        #W_z = tf.get_variable('W_z', shape=[dimension_H, dimension_H, dimension_H]nsion_H], stddev=0.1))
        #b_z = tf.get_variable('b_z', shape=[dimension_H, dimension_H]imension_H]))

        W_pre_omega = tf.get_variable('W_pre_omega', shape=[dimension_D, dimension_N])
        W_omega = tf.get_variable('W_omega', shape=[dimension_H, dimension_D])
        V_omega = tf.get_variable('V_omega', shape=[dimension_H, dimension_H])
        b_omega = tf.get_variable('b_omega', shape=[dimension_H])

    state_forget_gate = tf.sigmoid(tf.add(tf.add(tf.multiply(W_state, tf.reduce_sum(tf.multiply(W_pre_state,x),1,keep_dims=True)), tf.multiply(V_state, z_)), b_state), name='State_Forget')
    frequency_forget_gate = tf.sigmoid(tf.add(tf.add(tf.multiply(W_freq, tf.reduce_sum(tf.multiply(W_pre_freq,x),1,keep_dims=True)), tf.multiply(V_freq, z_)), b_freq), name='Freq_Forget')

    combined_forget_gate = tf.multiply(state_forget_gate, frequency_forget_gate)

    input_gate = tf.sigmoid(tf.add(tf.add(tf.multiply(W_g, tf.reduce_sum(tf.multiply(W_pre_g,x),1,keep_dims=True)), tf.multiply(V_g, z_)), b_g),name='Input_Gate')

    modulation_gate = tf.tanh(tf.add(tf.add(tf.multiply(W_i, tf.reduce_sum(tf.multiply(W_pre_i,x),1,keep_dims=True)), tf.multiply(V_i, z_)), b_i),name='Modulation_Gate')

    omega = tf.multiply(tf.add(tf.multiply(W_omega, tf.reduce_sum(tf.multiply(W_pre_omega,x),1,keep_dims=True)),tf.multiply(V_omega, z_)), b_omega)

    real_s = combined_forget_gate * Re_s_ + tf.multiply(input_gate * modulation_gate, tf.cos(omg_))
    img_s = combined_forget_gate * Im_s_ + tf.multiply(input_gate * modulation_gate, tf.sin(omg_))

    amplitude = tf.sqrt(real_s ** 2 + img_s ** 2)


    zz = 0
    """
    def __freq(inputs):
        U_k, W_k, V_k, b_k, W_z_k, b_z_k, A_k = tf.unstack(inputs)
        o = tf.sigmoid(tf.matmul(U_k, A_k) + tf.matmul(W_k, x) + tf.matmul(V_k, z) + b_k)

        zz = zz + o * tf.tanh(tf.matmul(training_params['W_z'][_], amplitude) + tf.transpose(training_params['b_z']))

        return zz

    _ = tf.scan(__freq,elems=[U_o,W_o,V_o,b_o,W_z,b_z,tf.transpose(amplitude)])
    """

    return tf.stack([omega, real_s, img_s, amplitude])

with tf.Session() as sess:

    #X, seq_labels = make_sequences(['AAPL'])

    X = np.random.rand(5, 5, 3).astype(np.float32)

    training_params = {}
    dimension_N = X.shape[2]
    dimension_H = X.shape[1]
    dimension_D = X.shape[1]
    num_frequencies = X.shape[2]
    batch_size = X.shape[0]

    x = tf.placeholder(tf.float32, shape=[batch_size, dimension_D, dimension_N])
    z_ = tf.constant(np.random.rand(dimension_H),dtype=tf.float32)

    build_graph()

    init = tf.global_variables_initializer()
    sess.run(init)
    pre_output = np.array([1.0,1.0,1.0,1.0]).astype(np.float32)

    #states = tf.scan(step,elems=X)


    for i in X:
        output = sess.run(step(pre_output,i))
        print(output[-1][0])