import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Get_Data import make_sequences

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
        self._inputs = tf.placeholder(tf.float64, shape=[None, self.state_size, self.input_size])
        self.ys = tf.placeholder(tf.float64, shape=[None, self.target_size])

        self.W_state = tf.Variable(np.random.uniform(size=[self.input_size, self.state_size]))
        self.V_state = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size]))
        self.b_state = tf.Variable(np.random.uniform(size=[self.state_size]))

        self.W_freq = tf.Variable(np.random.uniform(size=[self.input_size, self.state_size]))
        self.V_freq = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size]))
        self.b_freq = tf.Variable(np.random.uniform(size=[self.state_size]))

        self.W_g = tf.Variable(np.random.uniform(size=[self.input_size, self.state_size]))
        self.V_g = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size]))
        self.b_g = tf.Variable(np.random.uniform(size=[self.state_size]))

        self.W_i = tf.Variable(np.random.uniform(size=[self.input_size, self.state_size]))
        self.V_i = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size]))
        self.b_i = tf.Variable(np.random.uniform(size=[self.state_size]))

        self.U_o = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size, self.state_size]))
        self.W_o = tf.Variable(np.random.uniform(size=[self.state_size, self.input_size, self.state_size]))
        self.V_o = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size, self.state_size]))
        self.b_o = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size]))

        self.W_z = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size, self.state_size]))
        self.b_z = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size]))

        self.W_omega = tf.Variable(np.random.uniform(size=[self.input_size, self.state_size]))
        self.V_omega = tf.Variable(np.random.uniform(size=[self.state_size, self.state_size]))
        self.b_omega = tf.Variable(np.random.uniform(size=[self.state_size]))

        self.W_z_z = tf.Variable(np.random.uniform(size=[self.state_size, self.target_size]))
        self.b_z_z = tf.Variable(np.random.uniform(size=[self.target_size]))

        def step(prev_output, input):

            x_, t_ = input

            omg_, Re_s_, Im_s_, z_ = tf.unstack(prev_output)

            omg_ = omg_[:, :, 1]
            z_ = z_[:, :, 1]

            state_forget_gate = tf.sigmoid(tf.matmul(x_, self.W_state) + tf.matmul(z_, self.V_state) + self.b_state)
            frequency_forget_gate = tf.sigmoid(tf.matmul(x_,self.W_freq) + tf.matmul(z_, self.V_freq) + self.b_freq)

            state_forget_gate = tf.expand_dims(state_forget_gate,axis=-1)
            frequency_forget_gate = tf.expand_dims(frequency_forget_gate, axis=-1)
            frequency_forget_gate = tf.transpose(frequency_forget_gate,perm=[0,2,1])

            combined_forget_gate = tf.matmul(state_forget_gate, frequency_forget_gate)

            input_gate = tf.sigmoid(tf.matmul(x_, self.W_g) + tf.matmul(z_, self.V_g) + self.b_g)

            modulation_gate = tf.tanh(tf.matmul(x_, self.W_i) + tf.matmul(z_, self.V_i) + self.b_i)

            omega = tf.matmul(x_, self.W_omega) + tf.matmul(z_, self.V_omega) + self.b_omega

            real_c = tf.sin(omg_*t_)
            img_s = tf.sin(omg_ * t_)
            input_mod = input_gate * modulation_gate
            input_mod = tf.expand_dims(input_mod,axis=-1)

            real_c = tf.expand_dims(real_c,axis=-1)
            img_s = tf.expand_dims(img_s, axis=-1)
            real_c = tf.transpose(real_c,perm=[0,2,1])
            img_s = tf.transpose(img_s, perm=[0, 2, 1])

            real_s = combined_forget_gate * Re_s_ + tf.matmul(input_mod, real_c)
            img_s = combined_forget_gate * Im_s_ + tf.matmul(input_mod, img_s)

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
            new_z = tf.stack([new_z for _ in range(self.state_size)], axis=1)

            return tf.stack([omega, real_s, img_s, new_z])

        self.initial_hidden_time = tf.reduce_sum(self._inputs,2)
        self.initial_hidden_time = tf.expand_dims(self.initial_hidden_time,axis=-1)
        self.initial_hidden_time = tf.matmul(self.initial_hidden_time,tf.transpose(self.initial_hidden_time,perm=[0,2,1]))
        self.initial_hidden_time = tf.multiply(self.initial_hidden_time,tf.zeros([self.state_size, self.state_size],dtype=tf.float64))
        self.initial_hidden_time = self.initial_hidden_time + 1

        self.initial_hidden = tf.stack([self.initial_hidden_time, self.initial_hidden_time, self.initial_hidden_time, self.initial_hidden_time],axis=0)

        self.processed_input = tf.transpose(tf.transpose(self._inputs,perm=[2,0,1]))
        outputs = tf.scan(step, elems=[self.processed_input,(np.arange(self.state_size, dtype=np.float64) + 1) / self.state_size], initializer=self.initial_hidden)
        #last state of hidden state matrices
        last_state = tf.squeeze(outputs[-1:, -1:, :, -1:, :],axis=[0,1,3])
        #combine multifrequency states into one aggregate
        logits = tf.matmul(last_state, self.W_z_z) + self.b_z_z
        predictions = tf.nn.softmax(logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.ys,logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.ys, 1), tf.argmax(predictions, 1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

        self.predictions = predictions
        self.loss = loss
        self.train_op = train_op
        self.accuracy = accuracy

    def train(self, train_set, epochs=100, batch_size=100):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            train_accuracy = 0
            try:
                for i in range(epochs):
                    xs, ys = train_set
                    for offset in range(0, len(xs), batch_size):
                        batch_x = xs[offset: offset + batch_size]
                        batch_y = ys[offset: offset + batch_size]
                        _, train_loss_, accuracy = sess.run([self.train_op,self.loss, self.accuracy], feed_dict={self._inputs : batch_x, self.ys : batch_y})
                        train_loss += train_loss_
                        train_accuracy += accuracy
                    print('[{}] loss: {} accuracy: {}'.format(i,train_loss/100,train_accuracy/100))
                    train_loss = 0
                    train_accuracy = 0
            except KeyboardInterrupt:
                print('Interrupted by user')

X, Y = make_sequences(['AAPL'])

#X_train = np.random.rand(1000, 8, 20)
#y_train = np.random.choice([True,False],size=[1000, 2])

def get_on_hot(number):
    on_hot = [0] * 10
    on_hot[number] = 1
    return on_hot

digits = datasets.load_digits()
#X = digits.images
#Y_ = digits.target

#Y = [get_on_hot(x) for x in Y_]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.22)

# Cuttting for simple iteration
X_train = X_train[:1400]
y_train = y_train[:1400]

X_train = np.array(X_train)
y_train = np.array(y_train)

SFM = SFM_rnn(state_size=X_train.shape[1],input_size=X_train.shape[2], target_size=y_train.shape[1], model_name='SFM_1')
SFM.train(train_set=[X_train,y_train],epochs=10)