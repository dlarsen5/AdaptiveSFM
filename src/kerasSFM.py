from keras import activations
from keras import backend as K
from keras import constraints
from keras.utils import generic_utils
from keras import initializers
from keras.legacy import interfaces
from keras.engine.topology import Layer
from keras import regularizers
from keras.layers import recurrent

import numpy as np

import tensorflow as tf


class AdaSFMCell(Layer):
    '''
    Cell Class for the Adaptive SFM RNN Layer
    '''
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='zeros',
                 recurrent_initializer='zeros',
                 bias_initializer='ones',
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
        '''
        Parameters
        -------
        units: unsigned int
            - Dimensionality of the output space
        activation: func
            - Activation function to use (see [activations](../activations.md))
            - If you pass None, no activation is applied
              (ie. "linear" activation: `a(x) = x`)
        recurrent_activation: func
            - Activation function to use for the recurrent step
              (see [activations](../activations.md))
        use_bias: bool
            - Whether the layer uses a bias vector
        kernel_initializer: matrix
            - Initializer for the `kernel` weights matrix,
              used for the linear transformation of the inputs
              (see [initializers](../initializers.md))
        recurrent_initializer: matrix
            - Initializer for the `recurrent_kernel` weights matrix,
              used for the linear transformation of the recurrent state
              (see [initializers](../initializers.md))
        bias_initializer: matrix
            - Initializer for the bias vector
              (see [initializers](../initializers.md))
        unit_forget_bias: bool
            - If True, add 1 to the bias of the forget gate at initialization
            - Setting it to true will also force `bias_initializer="zeros"`
            - This is recommended in [Jozefowicz et al.]
              (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: func
            - Regularizer function applied to the `kernel` weights matrix
              (see [regularizer](../regularizers.md))
        recurrent_regularizer: func
            - Regularizer function applied to the `recurrent_kernel` weights matrix
              (see [regularizer](../regularizers.md))
        bias_regularizer: func
            - Regularizer function applied to the bias vector
              (see [regularizer](../regularizers.md))
        activity_regularizer: func
            - Regularizer function applied to the output
              of the layer (its "activation")
              (see [regularizer](../regularizers.md))
        kernel_constraint: func
            - Constraint function applied to the `kernel` weights matrix
              (see [constraints](../constraints.md))
        recurrent_constraint: func
            - Constraint function applied to the `recurrent_kernel` weights matrix
              (see [constraints](../constraints.md))
        bias_constraint: func
            - Constraint function applied to the bias vector
              (see [constraints](../constraints.md))
        dropout: float
            - Between 0 and 1
            - Fraction of the units to drop for
              the linear transformation of the inputs
        recurrent_dropout: float
            - Between 0 and 1.
            - Fraction of the units to drop for
              the linear transformation of the recurrent state
        implementation: unsigned int
            - Implementation mode
            - Either 1 or 2
        '''
        super(AdaSFMCell, self).__init__(**kwargs)
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
        # Input
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 5),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Recurrent Layer
        self.recur_k = self.add_weight(shape=(self.units, self.units * 5),
                                       name='recur_k',
                                       initializer=self.recurrent_initializer,
                                       regularizer=self.recurrent_regularizer,
                                       constraint=self.recurrent_constraint)
        # Fourier Frequency Layer
        self.freq_k = self.add_weight(shape=(self.units, self.units, self.units*3),
                                      name='freq_k',
                                      initializer=self.recurrent_initializer,
                                      regularizer=self.recurrent_regularizer,
                                      constraint=self.recurrent_constraint)
        # Fourier Frequency Layer Inputs
        self.freq_k_input = self.add_weight(shape=(self.units, input_dim, self.units),
                                            name='freq_k_input',
                                            initializer=self.recurrent_initializer,
                                            regularizer=self.recurrent_regularizer,
                                            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units), *args, **kwargs),
                        initializers.Ones()((self.units), *args, **kwargs),
                        self.bias_initializer((self.units * 2), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(shape=(self.units * 5),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            # Fourier Frequency Layer Bias
            self.freq_bias = self.add_weight(shape=(self.units*2,
                                                    self.units),
                                        name='freq_bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            self.freq_bias = None
        # State Layers
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_freq = self.kernel[:, self.units: self.units * 2]
        self.kernel_state = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_g = self.kernel[:, self.units * 3: self.units * 4]
        self.kernel_omg = self.kernel[:, self.units * 4:]
        # Recurrent Layers
        self.recur_k_i = self.recur_k[:, :self.units]
        self.recur_k_freq = self.recur_k[:, self.units: self.units * 2]
        self.recur_k_state = self.recur_k[:, self.units * 2: self.units * 3]
        self.recur_k_g = self.recur_k[:, self.units * 3: self.units * 4]
        self.recur_k_omg = self.recur_k[:, self.units * 4:]
        # Output Layers
        self.freq_k_W = self.freq_k_input
        self.freq_k_U = self.freq_k[:, :, :self.units]
        self.freq_k_V = self.freq_k[:, :, self.units: self.units * 2]
        self.freq_k_W_z = self.freq_k[:, :, self.units * 2:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_s = self.bias[self.units * 2: self.units * 3]
            self.bias_g = self.bias[self.units * 3: self.units * 4]
            self.bias_omg = self.bias[self.units * 4:]
            self.freq_bias_o = self.freq_bias[: self.units,:]
            self.freq_bias_z = self.freq_bias[self.units :,:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_s = None
            self.bias_g = None
            self.bias_omg = None
            self.freq_bias_z = None
            self.freq_bias_o = None

        self.built = True

    def call(self, inputs, states, training=None):
        '''
        Parameters
        -------
        inputs: 2D matrix
        states: 4D matrix
        training: bool

        Returns
        -------
        final_h: matrix
        final_state: 4D matrix
        '''
        # Dropout matrices for input units
        dp_mask = self._dropout_mask
        # Dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        z_ = states[0]
        Im_s_ = states[1]
        Re_s_ = states[2]
        omg_ = states[3]

        # Slice column from identical vectors
        omg_ = omg_[:, :, 1]
        z_ = z_[:, :, 1]

        inputs_ = inputs[0]
        t_ = inputs[1]

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
            x_g = K.bias_add(x_g, self.bias_g)
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

        freq = self.recurrent_activation(x_freq + K.dot(z_freq, self.recur_k_freq))
        state = self.recurrent_activation(x_state + K.dot(z_state, self.recur_k_state))
        fg = self._outer(freq, state)

        i = self.recurrent_activation(x_i + K.dot(z_i, self.recur_k_i))

        g = K.tanh(x_g + K.dot(z_g, self.recur_k_g))

        omega = x_omg + K.dot(z_omg, self.recur_k_omg)

        real_s = fg * Re_s_ + self._outer(i * g, K.cos(omg_ * t_))
        img_s = fg * Im_s_ + self._outer(i * g, K.sin(omg_ * t_))

        amplitude = K.sqrt(K.square(real_s) + K.square(img_s))
        # Transpose to dimensions (frequency_components, samples, state) for tf.scan
        amplitude = tf.transpose(amplitude, perm=[1, 0, 2])

        def __freq(z_k, inputs_):

            U_k, W_k, V_k, b_k, W_z_k, b_z_k, A_k = inputs_
            o = self.recurrent_activation(K.dot(A_k, U_k)
                                          + K.dot(inputs_o, W_k)
                                          + K.dot(z_o, V_k) + b_k)
            zz = z_k + o * K.tanh(K.dot(A_k, W_z_k) + b_z_k)

            return tf.stack(zz)

        h = tf.scan(__freq,
                    elems=[self.freq_k_U,self.freq_k_W,
                           self.freq_k_V,self.freq_bias_o,
                           self.freq_k_W_z,self.freq_bias_z,
                           amplitude],
                    initializer=tf.zeros(tf.shape(z_)))

        final_h = h[-1]

        # Fit size of other stacked matrices
        omega = tf.stack([omega for _ in range(self.state_size[0])], axis=1)
        final_h = tf.stack([final_h for _ in range(self.state_size[0])], axis=1)
        final_state = [final_h, img_s, real_s, omega]

        return final_h, final_state

    def _outer(self, x, y):
        '''
        Returns outer product between two two dimensional inputs

        Parameters
        -------
        x: 2D matrix
        y: 2D matrix

        Returns
        -------
        3D matrix
        '''
        return x[:, :, np.newaxis] * y[:, np.newaxis, :]

    def _generate_dropout_mask(self, inputs, training=None):
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._dropout_mask = [K.in_train_phase(dropped_inputs,
                                                   ones,
                                                   training=training)
                                  for _ in range(5)]
        else:
            self._dropout_mask = None

    def _generate_recurrent_dropout_mask(self, inputs, training=None):
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._recurrent_dropout_mask = [K.in_train_phase(dropped_inputs,
                                                             ones,
                                                             training=training)
                                            for _ in range(5)]
        else:
            self._recurrent_dropout_mask = None

class AdaSFM(recurrent.RNN):
    '''
    Adaptive State-Frequency Memory Recurrent Neural Network (Hu, Qi 2017.)

    References
    -------
    [State-Frequency Memory Recurrent Neural Networks] (http://proceedings.mlr.press/v70/hu17c/hu17c.pdf)
    [Learning to Forget: Continual Prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
    [Supervised Sequence Labeling with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
    [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='zeros',
                 recurrent_initializer='zeros',
                 bias_initializer='ones',
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
        '''
        Parameters
        -------
        units: unsigned int
            - Dimensionality of the output space
        activation: func
            - Activation function to use (see [activations](../activations.md)).
            - If you pass None, no activation is applied
              (ie. "linear" activation: `a(x) = x`)
        recurrent_activation: func
            - Activation function to use for the recurrent step
              (see [activations](../activations.md))
        use_bias: bool
            - Whether the layer uses a bias vector
        kernel_initializer: func
            - Initializer for the `kernel` weights matrix,
              used for the linear transformation of the inputs
              (see [initializers](../initializers.md))
        recurrent_initializer: func
            - Initializer for the `recurrent_kernel` weights matrix,
              used for the linear transformation of the recurrent state.
              (see [initializers](../initializers.md))
        bias_initializer: func
            - Initializer for the bias vector (see [initializers](../initializers.md))
        unit_forget_bias: bool
            - If True, add 1 to the bias of the forget gate at initialization
            - Setting it to true will also force `bias_initializer="zeros"`
            - This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: func
            - Regularizer function applied to the `kernel` weights matrix
              (see [regularizer](../regularizers.md))
        recurrent_regularizer: func
            - Regularizer function applied to the `recurrent_kernel` weights matrix
              (see [regularizer](../regularizers.md))
        bias_regularizer: func
            - Regularizer function applied to the bias vector
              (see [regularizer](../regularizers.md))
        activity_regularizer: func
            - Regularizer function applied to
              the output of the layer (its "activation")
              (see [regularizer](../regularizers.md))
        kernel_constraint: func
            - Constraint function applied to the `kernel` weights matrix
              (see [constraints](../constraints.md))
        recurrent_constraint:
            - Constraint function applied to the `recurrent_kernel` weights matrix
              (see [constraints](../constraints.md))
        bias_constraint: func
            - Constraint function applied to the bias vector
              (see [constraints](../constraints.md))
        dropout: float
            - Between 0 and 1
            - Fraction of the units to drop for
              the linear transformation of the inputs
        recurrent_dropout: float
            - Between 0 and 1
            - Fraction of the units to drop for
              the linear transformation of the recurrent state
        implementation: unsigned int
            - Implementation mode
            - Either 1 or 2
        return_sequences: bool
            - Whether to return the last output
              in the output sequence, or the full sequence
        return_state: bool
            - Whether to return the last state in addition to the output
        go_backwards: bool
            - (default False) If True, process the input sequence
              backwards and return the reversed sequence
        stateful: Boolean
            - (default False) If True, the last state
              for each sample at index i in a batch will be used as
              initial state for the sample of index i in the following batch
        '''
        cell = AdaSFMCell(units,
                                activation=activation,
                                recurrent_activation=recurrent_activation,
                                use_bias=use_bias,
                                kernel_initializer=kernel_initializer,
                                recurrent_initializer=recurrent_initializer,
                                unit_forget_bias=False,
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
        super(AdaSFM, self).__init__(cell,
                                           return_sequences=return_sequences,
                                           return_state=return_state,
                                           go_backwards=go_backwards,
                                           stateful=stateful,
                                           unroll=unroll,
                                           **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        '''
        Callable for one epoch of training

        Parameters
        -------
        inputs: tensor
        mask: tensor
        training: tensor
        initial_state: tensor
        constants: tensor

        Returns
        -------
        output: tensor
            - Sequences or last output based on `self.return_sequences==True`
        '''

        if isinstance(mask, list):
            mask = mask[0]

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

        fourier_timesteps = (np.arange(self.cell.state_size[0], dtype=np.float64) + 1) / self.cell.state_size[0]
        initial_state = initial_state[0]
        # Modified K.Rnn for adding Fourier timesteps dim
        last_output, outputs, states = self.adasfmrnn(self.cell.call,
                                                             inputs, fourier_timesteps,
                                                             initial_state,
                                                             constants=constants,
                                                             go_backwards=self.go_backwards,
                                                             mask=mask)

        outputs = tf.squeeze(outputs[:, :, :, -1:], axis=[-1])
        last_output = last_output[:, -1:, :]

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

    def adasfmrnn(self, step_function,
                         inputs, fourier_timesteps,
                         initial_states, go_backwards=False,
                         mask=None, constants=None):
        '''
        Adaptive SFM RNN iteration over the time dimension of a tensor

        Parameters
        -------
        step_function: RNN step function.
            - Parameters:
                * inputs: tensor with representing input for the
                  batch of samples at a certain time step
                    - Shape `(samples, ...)` (no time dimension)
                * states: list of tensors
            - Returns:
                * outputs: tensor with (no time dimension)
                * new_states: list of tensors, same length and shapes
                  as 'states'
                * The first state in the list must be the
                  output tensor at the previous timestep
                * Shape `(samples, output_dim)`
        inputs: tensor
            - (at least 3D) Tensor of temporal data
            - Shape `(samples, time, ...)`
        fourier_timesteps: numpy.array
            - Timesteps from n=1/timesteps to n=1
        initial_states: tensor
            - Containing the initial values for the states used in
              the step function
            - Shape (samples, output_dim) (no time dimension)
        go_backwards: bool
            - If True, do the iteration over the time
              dimension in reverse order and return the reversed sequence
        mask: tensor
            - Binary tensor with a zero for every element that is masked
            - Shape `(samples, time, 1)`
        constants: list
            - Constant values passed at each step

        Returns
        -------
        A tuple, `(last_output, outputs, new_states)`.

        last_output: tensor
            - The latest output of the rnn
            - Shape `(samples, ...)`
        outputs: tensor
            - Each entry `outputs[s, t]` is the output of the step
              function at time `t` for sample `s`
            - Shape `(samples, time, ...)`
        new_states: list of tensors
            - Latest states returned by the step function
            - Shape `(samples, ...)`

        Raises
        -------
        ValueError: if input dimension is less than 3.
        ValueError: if fourier timeseries length doesn't match time dimension of input
        ValueError: if `mask` is provided (not `None`) but states is not provided (`len(states)` == 0)
        '''
        ndim = len(inputs.get_shape())
        if ndim < 3:
            raise ValueError('Input should be at least 3D.')
        axes = [1, 0] + list(range(2, ndim))
        inputs = tf.transpose(inputs, (axes))

        if len(fourier_timesteps) != inputs.get_shape()[0]:
            raise ValueError("Fourier length doesn't match input time length")

        if mask is not None:
            if mask.dtype != tf.bool:
                mask = tf.cast(mask, tf.bool)
            if len(mask.get_shape()) == ndim - 1:
                mask = K.expand_dims(mask)
            mask = tf.transpose(mask, axes)

        if constants is None:
            constants = []

        global uses_learning_phase
        uses_learning_phase = False

        if not inputs.get_shape()[0]:
            raise ValueError('Unrolling requires a fixed number of timesteps.')
        states = initial_states
        successive_states = []
        successive_outputs = []

        input_list = tf.unstack(inputs)
        if go_backwards:
            input_list.reverse()
            np.flip(fourier_timesteps, axis=0)

        if mask is not None:
            mask_list = tf.unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for inp, mask_t in zip(zip(input_list, fourier_timesteps), mask_list):
                output, new_states = step_function(inp, states + constants)
                if getattr(output, '_uses_learning_phase', False):
                    uses_learning_phase = True

                '''
                Tiled Mask
                    - tf.where needs its condition tensor
                      to be the same shape as its two
                      result tensors, but in our case
                      the condition (mask) tensor is
                      (nsamples, 1), and A and B are (nsamples, ndimensions)
                    - So we need to broadcast the mask to match the shape of A and B.
                    - Tile call does this, it just repeats the mask along
                      its second dimension n times
                '''

                tiled_mask_t = tf.tile(mask_t,
                                       tf.stack([1, tf.shape(output)[1]]))

                if not successive_outputs:
                    prev_output = K.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = tf.where(tiled_mask_t, output, prev_output)

                return_states = []
                for state, new_state in zip(states, new_states):
                    # (see earlier comment for tile explanation)
                    tiled_mask_t = tf.tile(mask_t,
                                           tf.stack([1, tf.shape(new_state)[1]]))
                    return_states.append(tf.where(tiled_mask_t,
                                                  new_state,
                                                  state))
                states = return_states
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)
        else:
            for inp in zip(input_list, fourier_timesteps):
                output, states = step_function(inp, states)
                if getattr(output, '_uses_learning_phase', False):
                    uses_learning_phase = True
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)

        axes = [1, 0] + list(range(2, len(outputs.get_shape())))
        outputs = tf.transpose(outputs, axes)
        last_output._uses_learning_phase = uses_learning_phase
        return last_output, outputs, new_states

    def get_initial_state(self, inputs):
        '''
        Build initial state tensor
            - All-zero tensor
            - Fourier adds additional dim
            - Shape (samples, output_dim)
        Adaptive SFM initial state has shape (4, samples, state_size, state_size)
            - Inputs has dim (samples, timesteps, input_dim)

        Parameters
        -------
        inputs: tensor

        Returns
        -------
        list
        '''
        initial_state = K.zeros_like(inputs)
        initial_state = K.sum(initial_state, axis=(1, 2))
        initial_state = K.ones_like(initial_state)
        initial_state = K.expand_dims(initial_state)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.stack([self.cell._outer(K.tile(initial_state, [1, dim]), K.tile(initial_state, [1, dim])),
                             self.cell._outer(K.tile(initial_state, [1, dim]), K.tile(initial_state, [1, dim])),
                             self.cell._outer(K.tile(initial_state, [1, dim]), K.tile(initial_state, [1, dim])),
                             self.cell._outer(K.tile(initial_state, [1, dim]), K.tile(initial_state, [1, dim]))])
                    for dim in self.cell.state_size]
        else:
            return [K.stack([self.cell._outer(K.tile(initial_state, [1, self.cell.state_size]),
                                             K.tile(initial_state, [1, self.cell.state_size])),
                             self.cell._outer(K.tile(initial_state, [1, self.cell.state_size]),
                                             K.tile(initial_state, [1, self.cell.state_size])),
                             self.cell._outer(K.tile(initial_state, [1, self.cell.state_size]),
                                             K.tile(initial_state, [1, self.cell.state_size])),
                             self.cell._outer(K.tile(initial_state, [1, self.cell.state_size]),
                                             K.tile(initial_state, [1, self.cell.state_size]))])]

    @property
    def units(self):
        return self.cell.units

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
        base_config = super(AdaSFM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)

