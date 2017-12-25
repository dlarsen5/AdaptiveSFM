import tensorflow as tf
import numpy as np
from Get_Data import make_sequences

def step():
    #x, state_, z, seq_num = tf.unstack(inputs)
    #seq_num = tf.unstack(inputs)[0]

    with tf.name_scope('Cell'):

        Re_s_ = tf.get_variable('Re_s_',[1])
        Im_s_ = tf.get_variable('Im_s_',[1])
        omega_ = tf.get_variable('omega_',[1])

        z = tf.get_variable('z',shape=[sequence_length, sequence_length],initializer=tf.truncated_normal([sequence_length, sequence_length], stddev=0.1))

        zz = tf.get_variable('W_state',shape=[sequence_length],initializer=tf.truncated_normal([sequence_length], stddev=0.1))

        W_state = tf.get_variable('W_state',shape=[sequence_length, features],initializer=tf.truncated_normal([sequence_length,features], stddev=0.1))
        V_state = tf.get_variable('V_state',shape=[sequence_length, sequence_length],initializer=tf.truncated_normal([sequence_length,features], stddev=0.1))
        b_state = tf.get_variable('b_state',shape=[sequence_length],initializer=tf.constant(0.1, shape=[sequence_length]))

        W_freq =  tf.get_variable('W_freq',shape=[sequence_length, features],initializer=tf.truncated_normal([sequence_length,features], stddev=0.1))
        V_freq = tf.get_variable('V_freq', shape=[sequence_length, sequence_length],initializer=tf.truncated_normal([sequence_length, features], stddev=0.1))
        b_freq = tf.get_variable('b_freq',shape=[sequence_length],initializer=tf.constant(0.1, shape=[sequence_length]))

        W_g = tf.get_variable('W_g',shape=[sequence_length, features],initializer=tf.truncated_normal([sequence_length,features], stddev=0.1))
        V_g = tf.get_variable('V_g', shape=[sequence_length, sequence_length],initializer=tf.truncated_normal([sequence_length, features], stddev=0.1))
        b_g = tf.get_variable('b_g', shape=[sequence_length],initializer=tf.constant(0.1, shape=[sequence_length]))

        W_i = tf.get_variable('V_i',shape=[sequence_length, features],initializer=tf.truncated_normal([sequence_length,features], stddev=0.1))
        V_i = tf.get_variable('V_i', shape=[sequence_length, sequence_length],initializer=tf.truncated_normal([sequence_length, features], stddev=0.1))
        b_i = tf.get_variable('b_i',shape=[sequence_length],initializer=tf.constant(0.1, shape=[sequence_length]))

        U_o = tf.get_variable('U_o',shape=[sequence_length, features, sequence_length],initializer=tf.truncated_normal([sequence_length, features, sequence_length], stddev=0.1))
        W_o = tf.get_variable('W_o', shape=[sequence_length, features, features],initializer=tf.truncated_normal([sequence_length, features, sequence_length], stddev=0.1))
        V_o = tf.get_variable('V_o', shape=[sequence_length, features, sequence_length],initializer=tf.truncated_normal([sequence_length, features, sequence_length], stddev=0.1))
        b_o = tf.get_variable('b_i',shape=[sequence_length, features],initializer=tf.constant(0.1, shape=[sequence_length, features]))


        W_z = tf.get_variable('W_z', shape=[sequence_length, features, sequence_length],initializer=tf.truncated_normal([sequence_length, features], stddev=0.1))
        b_z = tf.get_variable('b_z', shape=[sequence_length, features], initializer=tf.constant(0.1, shape=[sequence_length]))

        W_omega = tf.get_variable('W_omega', shape=[sequence_length, features],initializer=tf.truncated_normal([sequence_length, sequence_length], stddev=0.1))
        V_omega = tf.get_variable('V_omega', shape=[sequence_length, sequence_length],initializer=tf.truncated_normal([sequence_length, sequence_length], stddev=0.1))
        b_omega = tf.get_variable('b_omega', shape=[sequence_length],initializer=tf.constant(0.1, shape=[sequence_length]))


    state_forget_gate = tf.sigmoid(tf.add(tf.add(tf.matmul(W_state[seq_num], x[seq_num]), tf.matmul(V_state[seq_num], z[seq_num])), b_state[seq_num]), name='State_Forget')
    frequency_forget_gate = tf.sigmoid(tf.add(tf.add(tf.matmul(W_freq[seq_num], x[seq_num]), tf.matmul(V_freq[seq_num], z[seq_num])), b_freq[seq_num]), name='Freq_Forget')

    combined_forget_gate = tf.multiply(state_forget_gate, frequency_forget_gate)

    input_gate = tf.sigmoid(tf.add(tf.add(tf.matmul(W_g[seq_num], x[seq_num]), tf.matmul(V_g[seq_num], z[seq_num])), b_g[seq_num]),name='Input_Gate')

    modulation_gate = tf.tanh(tf.add(tf.add(tf.matmul(W_i[seq_num], x[seq_num]), tf.matmul(V_i[seq_num], z[seq_num])), b_i[seq_num]),name='Modulation_Gate')

    omega = tf.matmul(tf.add(tf.add(W_omega[seq_num], x[seq_num]), tf.matmul(V_omega[seq_num], z[seq_num])), b_omega[seq_num])

    real_s = combined_forget_gate * Re_s_ + tf.multiply(input_gate * modulation_gate, tf.cos(omega_))
    img_s = combined_forget_gate * Im_s_ + tf.multiply(input_gate * modulation_gate, tf.sin(omega_))

    amplitude = tf.sqrt(real_s ** 2 + img_s ** 2)



    """
    zz = 0
    def __freq(inputs):
        U_k, W_k, V_k, b_k, W_z_k, b_z_k, A_k = tf.unstack(inputs)
        o = tf.sigmoid(tf.matmul(U_k, A_k) + tf.matmul(W_k, x) + tf.matmul(V_k, z) + b_k)

        zz = zz + o * tf.tanh(tf.matmul(training_params['W_z'][_], amplitude) + tf.transpose(training_params['b_z']))

        return zz

    _ = tf.scan(__freq,elems=[U_o,W_o,V_o,b_o,W_z,b_z,tf.transpose(amplitude)])"""

    return amplitude



with tf.Session() as sess:

    X, seq_labels = make_sequences(['AAPL'])

    training_params = {}
    features = X.shape[2]
    sequence_length = X.shape[1]
    num_frequencies = X.shape[2]
    input_dim = sequence_length

    x = tf.placeholder(tf.float32, shape=[sequence_length, features])

    init = tf.global_variables_initializer()

    sess.run(init)

    seq_num = tf.Variable(0,name='seq_num')

    for observation in X:
        for i in range(0,sequence_length-1):
            seq_num.assign(i)
            #inputs = tf.stack([seq_num])
            #output = tf.scan(step,name='step')
            observation = np.array(observation)
            run = sess.run(step,feed_dict={x : observation, seq_num : seq_num})
