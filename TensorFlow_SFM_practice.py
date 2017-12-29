import tensorflow as tf
import numpy as np
from Get_Data import make_sequences


def build_graph():
    #TODO build out computation graph with losses and gradients
    x_ = tf.placeholder(tf.float32, shape=[dimension_D, dimension_N])
    y_ = tf.placeholder(tf.float32, shape=[dimension_D])

    with tf.variable_scope('Cell'):
        W_state = tf.get_variable('W_state', shape=[dimension_D, dimension_N])
        V_state = tf.get_variable('V_state', shape=[dimension_H, dimension_H])
        b_state = tf.get_variable('b_state', shape=[dimension_H])

        W_freq = tf.get_variable('W_freq', shape=[dimension_D, dimension_N])
        V_freq = tf.get_variable('V_freq', shape=[dimension_H, dimension_H])
        b_freq = tf.get_variable('b_freq', shape=[dimension_H])

        W_g = tf.get_variable('W_g', shape=[dimension_D, dimension_N])
        V_g = tf.get_variable('V_g', shape=[dimension_H, dimension_H])
        b_g = tf.get_variable('b_g', shape=[dimension_H])

        W_i = tf.get_variable('W_i', shape=[dimension_D, dimension_N])
        V_i = tf.get_variable('V_i', shape=[dimension_H, dimension_H])
        b_i = tf.get_variable('b_i', shape=[dimension_H])

        U_o = tf.get_variable('U_o', shape=[dimension_H, dimension_H, dimension_H])
        W_o = tf.get_variable('W_o', shape=[dimension_H, dimension_D, dimension_N])
        V_o = tf.get_variable('V_o', shape=[dimension_H, dimension_H, dimension_H])
        b_o = tf.get_variable('b_o', shape=[dimension_H, dimension_H])


        W_z = tf.get_variable('W_z', shape=[dimension_H, dimension_H, dimension_H])
        b_z = tf.get_variable('b_z', shape=[dimension_H, dimension_H])

        W_omega = tf.get_variable('W_omega', shape=[dimension_D, dimension_N])
        V_omega = tf.get_variable('V_omega', shape=[dimension_H, dimension_H])
        b_omega = tf.get_variable('b_omega', shape=[dimension_H])

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
        new_z = tf.reduce_sum(new_z,0)


        return tf.stack([omega, real_s, img_s, new_z])

    return

def AdaptiveSFM(x):

    def step(prev_output, x_):

        omg_, Re_s_, Im_s_, z_ = tf.unstack(prev_output)

        with tf.variable_scope('Cell', reuse=True):
            W_state = tf.get_variable('W_state', shape=[dimension_D, dimension_N])
            V_state = tf.get_variable('V_state', shape=[dimension_H, dimension_H])
            b_state = tf.get_variable('b_state', shape=[dimension_H])

            W_freq = tf.get_variable('W_freq', shape=[dimension_D, dimension_N])
            V_freq = tf.get_variable('V_freq', shape=[dimension_H, dimension_H])
            b_freq = tf.get_variable('b_freq', shape=[dimension_H])

            W_g = tf.get_variable('W_g', shape=[dimension_D, dimension_N])
            V_g = tf.get_variable('V_g', shape=[dimension_H, dimension_H])
            b_g = tf.get_variable('b_g', shape=[dimension_H])

            W_i = tf.get_variable('W_i', shape=[dimension_D, dimension_N])
            V_i = tf.get_variable('V_i', shape=[dimension_H, dimension_H])
            b_i = tf.get_variable('b_i', shape=[dimension_H])

            U_o = tf.get_variable('U_o', shape=[dimension_H, dimension_H, dimension_H])
            W_o = tf.get_variable('W_o', shape=[dimension_H, dimension_D, dimension_N])
            V_o = tf.get_variable('V_o', shape=[dimension_H, dimension_H, dimension_H])
            b_o = tf.get_variable('b_o', shape=[dimension_H, dimension_H])

            W_z = tf.get_variable('W_z', shape=[dimension_H, dimension_H, dimension_H])
            b_z = tf.get_variable('b_z', shape=[dimension_H, dimension_H])

            W_omega = tf.get_variable('W_omega', shape=[dimension_D, dimension_N])
            V_omega = tf.get_variable('V_omega', shape=[dimension_H, dimension_H])
            b_omega = tf.get_variable('b_omega', shape=[dimension_H])

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
        new_z = tf.reduce_sum(new_z,0)


        return tf.stack([omega, real_s, img_s, new_z])

    outputs = tf.scan(step, elems=X, initializer=initial_weights)

    states = outputs[:,-1:,:,:]

    states = tf.squeeze(states, axis=1)
    states = tf.reduce_sum(states, axis=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=seq_labels, logits=states)

    print(loss)

    return states

with tf.Session() as sess:
    X, seq_labels = make_sequences(['AAPL'])

    #X = np.random.rand(5, 5, 3).astype(np.float32)

    X = X.astype(np.float32)

    training_params = {}
    dimension_N = X.shape[2]
    dimension_H = X.shape[1]
    dimension_D = X.shape[1]
    num_frequencies = X.shape[2]
    batch_size = X.shape[0]



    build_graph()

    init = tf.global_variables_initializer()
    sess.run(init)
    initial_weights = np.ones((4, X.shape[1], X.shape[1]), dtype=np.float32)

    states = sess.run(AdaptiveSFM(X))