__author__ = 'Rakesh R Menon'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _mlp(input, output_size, hidden_layers, init_weights=[], activation_fn=None, norm=None, scope_name="default"):

	'''
		Function : Function for constructing MLPs

		Inputs:

			input [tf_placeholder]: The input data placeholder with the dimension required to form the MLP.
			output_size [Int]: The final output number of neurons of the MLP.
			hidden_layers [List]: A list with number of elements equal to the number of hidden layers. Each element in the list would be the number of neurons in each hidden layer.
			init_weights [List]: A list with the weight initializations that needs to be applied to each layer. If empty list, then Xavier initialization is applied.
			activation_fn [String]: A string mentioning the kind of activation function that will be used in all the hidden layers of the MLP. 
			norm [String]: A string accepting the kind of normalization that is required to be applied between two layers of the MLP. Available options: ["batch", "layer", "None"]
			scope_name [String]: Tensorflow scope name

		Output:

			net : A tf_layers module
	'''

	if init_weights==[]:
		init_weights = [tf.contrib.layers.xavier_initializer()]*(len(hidden_layers)+1)		# +1 for the output layer initialization

	with tf.variable_scope(scope_name):
		net = input
		for i in range(len(hidden_layers)):

			net = tf_layers.fully_connected(net, num_outputs=hidden_layers[i], activation_fn=None, weights_initializer=init_weights[i])
			if norm:
				net = _norms(net, norm)
			if isinstance(activation_fn, (list, tuple)):
				net = _activation(net, activation_fn[i])
			else:
				net = _activation(net, activation_fn)

		net = tf_layers.fully_connected(net, num_outputs=output_size, activation_fn=None, weights_initializer=init_weights[-1])

	return net

def _norms(input, norm):

	'''
		Function : To apply normalization.

		Inputs:

			input [tf tensor]: The input data placeholder/variable.
			norm [String]: A string mentioning the kind of normalization to be used. Currently supported ["batch", "layer"]
			scope_name [String]: Tensorflow scope name

		Output:

			net : tf_layers module
	'''
	if norm=="batch":
		net = tf_layers.batch_norm(input, activation_fn=None)
	elif norm=="layer":
		net = tf_layers.layer_norm(input, activation_fn=None, trainable=True)
	else:
		raise ValueError('Current implementation only uses batch normalization and layer normalization. Please give a valid input.')

	return net

def _activation(input, activation_fn):

	'''

		Function : To apply activations over neural network outputs.

		Inputs:

			input [tf tensor]: The input data placeholder/variable.
			norm [String]: A string mentioning the kind of normalization to be used. Currently supported ["relu", "sigmoid", "tanh", "swish", "elu", softmax"]
			scope_name [String]: Tensorflow scope name

		Output:

			net : tf_layers module
	'''

	if activation_fn=="relu":
		net = tf.nn.relu(input)
	elif activation_fn=="sigmoid":
		net = tf.nn.sigmoid(input)
	elif activation_fn=="tanh":
		net = tf.nn.tanh(input)
	elif activation_fn=="swish":
		net = tf.nn.sigmoid(input)*input
	elif activation_fn=="elu":
		net = tf.nn.elu(input)
	elif activation_fn=="softmax":
		net = tf.nn.softmax(input)
	elif activation_fn is None:
		net = input
	else:
		raise ValueError('Current implementation only uses relu, sigmoid, tanh, swish and softmax activations. Please give a valid input.')

	return net

def mlp(input, output_size, hidden_layers=[], *args, **kwargs):

	'''
		Function : Callable function for creating multi layer perceptrons

		Input:

			input [tf_placeholder]: The input data placeholder with the dimension required to form the MLP.
			output_size [Int]: The final output number of neurons of the MLP.
			hidden_layers [List]: A list with number of elements equal to the number of hidden layers. Each element in the list would be the number of neurons in each hidden layer.

		Output:

			net : tf_layers module
	'''
	
	net = _mlp(input, output_size, hidden_layers, *args, **kwargs)
	return net

def classifier(input, output_size, hidden_layers=[], activation_fn=None, final_activation_fn="softmax", normalization=None, scope="default", **kwargs):

	'''

		Function : To create a simple classifier network.

		Inputs:

			input [tf_placeholder]: The input data placeholder with the dimension required to form the MLP.
			output_size [Int]: The final output number of neurons of the MLP.
			hidden_layers [List]: A list with number of elements equal to the number of hidden layers. Each element in the list would be the number of neurons in each hidden layer.
			activation_fn [String]: A string mentioning the kind of activation function that will be used in all the hidden layers of the MLP. 
			norm [String]: A string accepting the kind of normalization that is required to be applied between two layers of the MLP. Available options: ["batch", "layer", "None"]
			scope_name [String]: Tensorflow scope name

		Output:

			net : tf_layer module
	'''

	net = mlp(input, output_size, hidden_layers, activation_fn=activation_fn, norm=normalization, scope=scope, **kwargs)
	net = _activation(net, final_activation_fn, scope_name=scope)
	return net

def _create_conv_weights(height, width, inp_channels, out_channels, initializer):

	'''
		Function : Create convolutional layer weights

		Inputs:

			height [Int]: height of filter
			width [Int]:  width of filter
			inp_channels [Int]: number of input channels
			out_channels [Int]:	number of output channels
			initializer [String]: initializer to be used. Currently handles ["xavier", "random", "zeros"]

		Outputs:

			w : weights
			b : biases
	'''

	if initializer=="xavier":
		w = tf.Variable(tf.contrib.layers.xavier_initializer([height, width, inp_channels, out_channels]))
		b = tf.Variable(tf.contrib.layers.xavier_initializer([out_channels]))
	elif initializer=="random":
		w = tf.Variable(tf.random.normal([height, width, inp_channels, out_channels]))
		b = tf.Variable(tf.random.normal([out_channels]))
	else:
		w = tf.Variable(tf.contrib.layers.xavier_initializer([height, width, inp_channels, out_channels]))
		b = tf.Variable(tf.zeros([out_channels]))

	return w,b

def conv_layer(input, filter_sizes, out_channels, strides, padding=None, initializer=None, norms=None, activation_fn=None, max_pooling=None, scope="default"):

	'''
		Function : Create convolutional network

		Inputs:

			input [tf_placeholder/tf_variable] : Input for convolutional layer
			filter_size [List]: filter sizes for different layers. [only square filters are compatible]
			inp_channels [List]: 	number of input channels
			out_channels [List]:	number of output channels
			stride [List] : List with strides
			padding [String/List]: padding to use. Options available ["SAME", "VALID"]
			initializer [List/tf initializer function]: initializer to be used. Currently handles ["xavier", "random", "zeros"]
			activation_fn [String/List]: A string mentioning the kind of activation function that will be used in all the hidden layers of the MLP. 
			norm [String/List]: A string accepting the kind of normalization that is required to be applied between two layers of the MLP. Available options: ["batch", "layer", "None"]
			scope [String]: name of scope to be used

		Outputs:

			net : tf_layer module
	'''

	if not isinstance(initializer, (list, tuple)):
		if isinstance(initializer, str):
			initializer = [initializer]*len(out_channels)
		else:
			initializer = [tf.contrib.layers.xavier_initializer()]*len(out_channels)

	if not isinstance(padding, (list, tuple)):
		if isinstance(padding, str):
			padding = [padding]*len(out_channels)
		else:
			padding = ["SAME"]*len(out_channels)

	if not isinstance(max_pooling, (list, tuple)):
		if isinstance(max_pooling, str):
			max_pooling = [max_pooling]*len(out_channels)
		else:
			max_pooling = [2]*len(out_channels)

	with tf.variable_scope(scope):
		x = input
		for i in range(len(out_channels)):
			x = conv2d(x, [filter_size[i], filter_size[i]], out_channels[i], [1, strides[i], strides[i], 1], padding[i], initializer[i], scope=scope)
			if norm:
				x = _norms(x, norm)
			if isinstance(activation_fn, (list, tuple)):
				x = _activation(x, activation_fn[i])
			else:
				x = _activation(x, activation_fn)
			x = max_pool(x, kernel=[max_pooling[i], max_pooling[i]])

	return x


def conv2d(input, filter_size, out_channels, stride, padding="SAME", initializer=tf.contrib.layers.xavier_initializer(), scope="default"):

	'''
		Function : Create convolutional layer

		Inputs:

			input [tf_placeholder/tf_variable] : Input for convolutional layer
			filter_size [List]: height and width of convolutional filter
			inp_channels [Int]: number of input channels
			out_channels [Int]:	number of output channels
			stride [List] : 1D tensor of length 4  with strides
			padding [String]: padding to use. Options available ["SAME", "VALID"]
			initializer [String]: initializer to be used. Currently handles ["xavier", "random", "zeros"]
			scope [String]: name of scope to be used

		Outputs:

			net : tf_layer module
	'''
	
	with tf.variable_scope(scope):
		w, b = _create_conv_weights(filter_size[0], filter_size[1], int(x.get_shape.as_list()[-1]), out_channels, initializer)
		net = tf.nn.conv2d(input, w, strides, padding) + b

	return net

def deconv2d(input, filter_size, out_channels, stride, padding="SAME", initializer=tf.contrib.layers.xavier_initializer(), scope="default"):

	'''
		Function : Create upconvolutional layer

		Inputs:

			input [tf_placeholder/tf_variable] : Input for convolutional layer
			filter_size [List]: height and width of convolutional filter
			inp_channels [Int]: number of input channels
			out_channels [Int]:	number of output channels
			stride [List] : 1D tensor of length 4 with strides
			padding [String]: padding to use. Options available ["SAME", "VALID"]
			initializer [String]: initializer to be used. Currently handles ["xavier", "random", "zeros"]
			scope [String]: name of scope to be used

		Outputs:

			net : tf_layer module
	'''
	
	with tf.variable_scope(scope):
		w, b = _create_conv_weights(filter_size[0], filter_size[1], out_channels, int(x.get_shape.as_list()[-1]), initializer)
		net = tf.nn.conv2d_transpose(input, w, strides, padding) + b

	return net

def max_pool(input, kernel=[2, 2], strides=[2, 2]):
	'''

		Function : max_pooling layer

		Inputs:
			input [tf tensor/placeholder]: A 4D tensor
			kernel [List]: 2D or 4D list indicating the kernel size.
			strides [List]: 2D or 4D list indicating the strides.

		Outputs:

			out [tf tensor]: max-pooled output
	'''

	if len(kernel)==2:
		kernel = [1, kernel[0], kernel[1], 1]

	if len(strides)==2:
		strides = [1, strides[0], strides[1], 1]

	return tf.nn.max_pool(input, ksize=kernel, strides=strides, padding="SAME")


def trainable_vars(scope_name="default"):
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


def global_vars(scope_name="default"):
	return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

'''

	Taken from https://github.com/tensorflow/models/blob/master/research/pcl_rl/trust_region.py
'''
def gradient(loss, vars):
	grad = tf.gradients(loss, vars)
	return [g if g is not None else tf.zeros_like(v)
	 for (v,g) in zip(vars, grad)]

'''

	Taken from https://github.com/tensorflow/models/blob/master/research/pcl_rl/trust_region.py
'''
def flatgrad(loss, vars):

	grad = gradient(loss, vars)
	return tf.concat([tf.reshape(grad, [-1]) for g in grad], axis=0)

def flatten(input):
	'''

		Function : flatten any n-dimensional tensor to a 2D tensor. (n>=2)

		Inputs:

			input [tf variable/placeholder] : n-dimensional tensor

		Outputs:
			
			out [tf variable/tensor] : 2-dimensional tensor
	'''

	input_shape = input.get_shape().as_list()
	return tf.reshape(x, [-1, np.prod(input_shape[1:])])

def update_vars_op(scope_from, scope_to):
	'''

		Function : Create operations for assigning values of variables from one scope to another.

		Inputs:

			scope_from [String]: Scope of variables from which to assign
			scope_to [String]: Scope of variables that need to be assigned

		Output:

			ops [List]: List of tensorflow operations that needs to be carried out to assign.
	'''

	vars_from = trainable_vars(scope_from)
	vars_to = trainable_vars(scope_to)

	ops = []
	for (vf, vt) in zip(vars_from, vars_to):
		ops.append(tf.assign(vt, vf))

	return ops

def lstm(hidden_size, forget_bias=0.0, state_is_tuple=True, mode="BASIC", scope="default"):
	'''

		Function : Tensorflow LSTM wrapper

		Inputs:

			hidden_size [Int]: The hidden layer size of the LSTM.
			forget_bias [Float]: Additional bias implemented to vary forget gate effect.
			state_is_tuple [Bool]: Whether the state needs to be output as a tuple or a concatenated tensor
			mode [String]: BASIC or BLOCK LSTM. Refer to documentation for details.
			scope [String]: scope name.

		Outputs:

			_lstm [tensorflow module]
	'''
	
	with tf.variable_scope(scope):
		if mode=="BASIC":
			_lstm = tf.contrib.rnn.BasicLSTMCell(
				hidden_size, forget_bias=forget_bias, state_is_tuple=state_is_tuple)
		elif mode=="BLOCK":
			_lstm = tf.contrib.rnn.LSTMBlockCell(
				hidden_size, forget_bias=forget_bias)
		else:
			raise ValueError('Invalid LSTM mode provided. Please use one of BLOCK or BASIC!')

	return _lstm
