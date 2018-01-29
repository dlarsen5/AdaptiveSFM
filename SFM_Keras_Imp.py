from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from numpy import newaxis
from keras import activations
from keras.layers import recurrent
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import generic_utils
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.legacy import interfaces

import Get_Data

X,Y = Get_Data.make_digits_data()
#X,Y = Get_Data.make_bit_sequences('btc')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.22)

#X_train = X_train[:2000]
#y_train = y_train[:2000]

X_train = np.array(X_train).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

class SFMCell(Layer):
    """Cell class for the LSTM layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(SFMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        #change second dim to 5
        self.kernel = self.add_weight(shape=(input_dim, self.units * 5),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 5),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.frequency_kernel = self.add_weight(shape=(self.units,self.units,self.units*3),
                                                name='frequency_kernel',
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint
                                                )
        self.frequency_kernel_input = self.add_weight(shape=(self.units, input_dim, self.units),
                                                name='frequency_kernel_input',
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint
                                                )

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 5,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.freq_bias = self.add_weight(shape=(self.units,self.units * 2,),
                                        name='freq_bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            self.freq_bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_freq = self.kernel[:, self.units: self.units * 2]
        self.kernel_state = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_g = self.kernel[:, self.units * 3: self.units * 4]
        self.kernel_omg = self.kernel[:, self.units * 4:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_freq = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_state = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_g = self.recurrent_kernel[:, self.units * 3: self.units * 4]
        self.recurrent_kernel_omg = self.recurrent_kernel[:, self.units * 4:]

        self.frequency_kernel_W = self.frequency_kernel_input
        self.frequency_kernel_U = self.frequency_kernel[:, :, :self.units]
        self.frequency_kernel_V = self.frequency_kernel[:, :, self.units: self.units * 2]
        self.frequency_kernel_W_z = self.frequency_kernel[:, :, self.units * 2:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_s = self.bias[self.units * 2: self.units * 3]
            self.bias_g = self.bias[self.units * 3: self.units * 4]
            self.bias_omg = self.bias[self.units * 4:]

            self.freq_bias_o = self.freq_bias[: self.units]
            self.freq_bias_z = self.freq_bias[self.units :]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_s = None
            self.bias_c = None
            self.bias_o = None
            self.bias_omg = None
            self.freq_bias_z = None
            self.freq_bias_o = None
        self.built = True

    def _generate_dropout_mask(self, inputs, training=None):
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(6)]
        else:
            self._dropout_mask = None

    def _generate_recurrent_dropout_mask(self, inputs, training=None):
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._recurrent_dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(6)]
        else:
            self._recurrent_dropout_mask = None

    def outer_product(self,inputs):
        x, y = inputs
        outerProduct = x[:, :, newaxis] * y[:, newaxis, :]
        #batchSize = K.shape(x)[0]
        #outerProduct = K.reshape(outerProduct, (batchSize, -1))
        return outerProduct

    def call(self, inputs, states, training=None):
        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        Im_s_ = states[0]
        Re_s_ = states[1]
        z_ = states[2]
        omg_ = states[3]

        omg_ = omg_[:, :, 1]
        z_ = z_[:, :, 1]

        t_ = inputs[1]
        inputs_ = inputs[0]

        if 0. < self.dropout < 1.:
            inputs_i = inputs_* dp_mask[0]
            inputs_state = inputs_* dp_mask[1]
            inputs_freq = inputs_* dp_mask[2]
            inputs_g = inputs_* dp_mask[3]
            inputs_omg = inputs_* dp_mask[4]
            inputs_o = inputs_* dp_mask[5]
        else:
            inputs_i = inputs_
            inputs_freq = inputs_
            inputs_state = inputs_
            inputs_g = inputs_
            inputs_omg = inputs_
            inputs_o = inputs_

        x_i = K.dot(inputs_i, self.kernel_i)
        x_freq = K.dot(inputs_freq, self.kernel_freq)
        x_state = K.dot(inputs_state, self.kernel_state)
        x_g = K.dot(inputs_g, self.kernel_g)
        x_omg = K.dot(inputs_omg, self.kernel_omg)

        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias_i)
            x_freq = K.bias_add(x_freq, self.bias_f)
            x_state = K.bias_add(x_state, self.bias_s)
            x_g = K.bias_add(x_g, self.bias_c)
            x_omg = K.bias_add(x_omg, self.bias_omg)

        if 0. < self.recurrent_dropout < 1.:
            z_i = z_ * rec_dp_mask[0]
            z_freq = z_ * rec_dp_mask[1]
            z_state = z_ * rec_dp_mask[2]
            z_g = z_ * rec_dp_mask[3]
            z_omg = z_ * rec_dp_mask[4]
            z_o = z_ * rec_dp_mask[5]
        else:
            z_i = z_
            z_freq = z_
            z_state = z_
            z_g = z_
            z_omg = z_
            z_o = z_

        freq = self.recurrent_activation(x_freq + K.dot(z_freq, self.recurrent_kernel_freq))
        state = self.recurrent_activation(x_state + K.dot(z_state, self.recurrent_kernel_state))
        combined_forget_gate = self.outer_product([freq, state])

        i = self.recurrent_activation(x_i + K.dot(z_i, self.recurrent_kernel_i))

        g = K.tanh(x_g + K.dot(z_g, self.recurrent_kernel_g))

        omega = x_omg + K.dot(z_omg, self.recurrent_kernel_omg)

        real_s = combined_forget_gate * Re_s_ + self.outer_product([i * g, K.cos(omg_ * t_)])
        img_s = combined_forget_gate * Im_s_ + self.outer_product([i * g, K.sin(omg_ * t_)])

        amplitude = K.sqrt(K.square(real_s) + K.square(img_s))

        def __freq(z_k, states):

            U_k, W_k, V_k, b_k, W_z_k, b_z_k, A_k = states
            o = self.recurrent_activation(K.dot(A_k, U_k) + K.dot(inputs_o, W_k) + K.dot(z_o, V_k) + b_k)
            zz = z_k + o * K.tanh(K.dot(A_k, W_z_k) + b_z_k)

            return zz

        h, outputs, states = K.rnn(__freq,inputs=[self.frequency_kernel_U,self.frequency_kernel_W,
                                     self.frequency_kernel_V,self.freq_bias_o,
                                     self.frequency_kernel_W_z,self.freq_bias_z,
                                     K.transpose(amplitude)],initial_states=K.zeros(shape=K.shape(z_),dtype='float64'))

        omega = tf.stack([omega for _ in range(self.state_size[0])], axis=1)
        h = tf.stack([h for _ in range(self.state_size[0])], axis=1)

        return h, [img_s, real_s, h, omega]

class SFM(recurrent.RNN):
    """Long-Short Term Memory layer - Hochreiter 1997.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):

        cell = SFMCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        super(SFM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    @property
    def states(self):
        if self._states is None:
            if isinstance(self.cell.state_size, int):
                num_states = 1
            else:
                num_states = len(self.cell.state_size)
            return [None for _ in range(num_states)]
        return self._states

    @states.setter
    def states(self, states):
        self._states = states

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.stack([self.outer_product([K.tile(initial_state, [1, dim]), K.tile(initial_state, [1, dim])]),
                             self.outer_product([K.tile(initial_state, [1, dim]), K.tile(initial_state, [1, dim])]),
                             self.outer_product([K.tile(initial_state, [1, dim]), K.tile(initial_state, [1, dim])]),
                             self.outer_product([K.tile(initial_state, [1, dim]), K.tile(initial_state, [1, dim])])])
                    for dim in self.cell.state_size]
        else:
            return [K.stack([self.outer_product([K.tile(initial_state, [1, self.cell.state_size]), K.tile(initial_state, [1, self.cell.state_size])]),
                             self.outer_product([K.tile(initial_state, [1, self.cell.state_size]), K.tile(initial_state, [1, self.cell.state_size])]),
                             self.outer_product([K.tile(initial_state, [1, self.cell.state_size]), K.tile(initial_state, [1, self.cell.state_size])]),
                             self.outer_product([K.tile(initial_state, [1, self.cell.state_size]), K.tile(initial_state, [1, self.cell.state_size])])])]

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):

        if isinstance(mask, list):
            mask = mask[0]

        input_shape = inputs.shape
        timesteps = input_shape[1]

        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        kwargs = {}
        if generic_utils.has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        self.cell._generate_dropout_mask(inputs, training=training)
        self.cell._generate_recurrent_dropout_mask(inputs, training=training)

        inputs = tf.transpose(tf.transpose(inputs,perm=[2,0,1]))

        fourier_timesteps = (np.arange(self.cell.state_size[0], dtype=np.float64) + 1) / self.cell.state_size[0]
        #todo K.rnn needs a tensor as input for get_shape() function
        last_output, outputs, states = K.rnn(self.cell.call,
                                             [inputs,fourier_timesteps],
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             unroll=self.unroll,
                                             input_length=timesteps)

        outputs = tf.squeeze(outputs[:, -1:, :, :, -1:], axis=[1, 4])

        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(SFM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


mhoy = SFM(X_train.shape[1],input_shape=(X_train.shape[1],X_train.shape[2]))

mhoy.call(inputs=X_train)