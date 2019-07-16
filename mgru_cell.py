import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class MgRUCell(tf.nn.rnn_cell.RNNCell):
    """ ref: tf.contrib.rnn.LSTMCell
    """

    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None):
        super(MgRUCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, _) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:
            if self._num_unit_shards is not None:
                unit_scope.set_partitioner(
                    partitioned_variables.fixed_size_partitioner(
                        self._num_unit_shards))
            mgru_matrix = _linear([inputs, c_prev], 2 * self._num_units, bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            f, o = array_ops.split(value=mgru_matrix, num_or_size_splits=2, axis=1)

            input_size = inputs.get_shape().as_list()[-1]
            with vs.variable_scope("input_projection"):
                if input_size == self._num_units:
                    input_proj = inputs
                else:
                    input_proj = _linear(inputs, self._num_units, bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
                    input_proj = self._activation(input_proj)

            with vs.variable_scope("c_bar"):
                c_ = _linear(inputs, self._num_units, bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
                c_ = self._activation(c_)

            c = (1-sigmoid(f)) * c_prev + sigmoid(f)*c_
            m = (1-sigmoid(o)) * c + sigmoid(o)*input_proj

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else array_ops.concat([c, m], 1))
        return m, new_state

def _linear(args, output_size, bias, bias_initializer=None, kernel_initializer=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(_WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype,
                                  initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(_BIAS_VARIABLE_NAME, [output_size], dtype=dtype, initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)
