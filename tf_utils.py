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

def trainable_vars(scope_name="default"):
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)

def global_vars(scope_name="default"):
	return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

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
