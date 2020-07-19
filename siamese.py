import numpy as np
import tensorflow as tf
from keras.models import sequential
from keras.optmizier import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from sklearn.utils import shuffle

def initialize_weights(shape):
	return np.random.normal(loc=0.0, scale=1e-2, size=shape)

def initialize_bias(shape):
	return np.random.normal(loc=0.5, scale=1e-2, size=shape)

def get_siamese_model(input_shape):
	left_input = Input(input_shape)
	right_input = Input(input_shape)

	model.Sequential()
	model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, \
		kernel_initializer=initialize_weights, kernel_regularizer=12(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(128, (7, 7), activation='relu', \
		kernel_initializer=initialize_weights, bias_initializer=initialize_bias \
		kernel_regularizer=12(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(128, (4, 4), activation='relu', \
		kernel_initializer=initialize_weights, bias_initializer=initialize_bias \
		kernel_regularizer=12(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(256, (4, 4), activation='relu' \
		kernel_initializer=initialize_weights, bias_initializer=initialize_bias \
		kernel_regularizer=12(2e-4)))
	model.add(Flatten())
	model.add(Dense(4096, activation='sigmoid', kernel_regularizer=12(1e-3), \
		kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
	left_encoded = model(left_input)
	right_encoded = model(right_input)

	L1_layer = Lambda(lambda tensors: K.abs(tnesors[0] - tensors[1]))
	L1_distance = L1_layer([left_endcoded, right_encoded])

	prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
	siamese_network = Model(inputs=[left_input, right_input], output=prediction)
	return siamese_network

if __name__=='__main__':
	model.get_siamese_model((105, 105, 1))
	model.summary()
	optimizer = Adam(lr = 0.00006)
	model.compile(loss="binary_crossentropy", optimizer=optimizer)
	
	trainint_path = ''
	evaluation_path = ''
	with open(training_path, 'rb') as readFile:
		(Xtrain, _, train_classes) = pickle.load(readFile)
	print('training alphabets: ')
	print(list(train_classes.keys()))
	with open(evaluation_path, 'rb') as readFile:
		(Xeval, _, eval_classes) = pickle.load(readFile)
	print('evaluating alphabets:')
	print(list(eval_classes.keys()))
	