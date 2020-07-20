import numpy as np
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform, RandomNormal
from keras.models import Model

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K


from batch_generator import BatchManager
def get_siamese_model(input_shape):
	left_input = Input(input_shape)
	right_input = Input(input_shape)

	model = Sequential()
	model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, \
		 kernel_regularizer=l2(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(128, (7, 7), activation='relu', \
		kernel_regularizer=l2(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(128, (4, 4), activation='relu', \
		kernel_regularizer=l2(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(256, (4, 4), activation='relu', \
		kernel_regularizer=l2(2e-4)))
	model.add(Flatten())
	model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3) ))
	left_encoded = model(left_input)
	right_encoded = model(right_input)

	L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
	L1_distance = L1_layer([left_encoded, right_encoded])

	prediction = Dense(1, activation='sigmoid')(L1_distance)
	siamese_network = Model(inputs=[left_input, right_input], output=prediction)
	return siamese_network

def test_oneshot_once(model, validation_batch, N, verbose=True):
	pairs, targets = validation_batch
	if targets.shape[0] < N:
		raise ValueError('supported images are : {} which is less than \
			Classification Way: {}'.format(pairs[1].shape, N))
# 	if verbose:
# 		print('Evaluating model on validation set with random {} way one-shot learning tasks...'.format(N))
	probabilities = model.predict(pairs)
	if np.argmax(probabilities) == np.argmax(targets):
		return True
	else: return False

def test_model(model, batch_manager, N, epochs, batch_size, verbose=True):
	true_positive = 0
	for i in range(epochs):
# 		print('Evaluating model on random vevaluation set of batch_size {}, in {} way classification'.format( \
# 			batch_size, N))
		validation_batch = batch_manager.get_test_batch(batch_size=batch_size)
		if test_oneshot_once(model, validation_batch, N) is True:
			true_positive += 1
		if verbose: print('[evaluating] : Accuracy: {}%  - {} way one shot learning'.format((100 * true_positive)/(i+1), N))
	percent = (100 * true_positive)/epochs
	print('Accuracy: {}%  - {} way one shot learning'.format(percent, N))
	return percent

if __name__=='__main__':
	siamese = get_siamese_model((105, 105, 1))
	siamese.summary()
	optimizer = Adam(lr = 0.00006)
	siamese.compile(loss="binary_crossentropy", optimizer=optimizer)
	
	training_path = './../fellowship.ai/omniglot/python/training.pkl'
	evaluation_path = './../fellowship.ai/omniglot/python/evaluating.pkl'
	with open(training_path, 'rb') as readFile:
		(Xtrain, _, train_classes) = pickle.load(readFile)
	print('training alphabets: ')
	print(list(train_classes.keys()))
	with open(evaluation_path, 'rb') as readFile:
		(Xeval, _, eval_classes) = pickle.load(readFile)
	print('evaluating alphabets:')
	print(list(eval_classes.keys()))
	