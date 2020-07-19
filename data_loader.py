import os
import numpy as np
import imageio
import pickle

class DataLoader:
	def __init__(self, data_path, batch_size=10, image_width=105, image_height=105):
		self.data_path = data_path
		self.train_paths_dictionary = {}
		self.evaluation_paths_dictionary = {}
		self.image_width = image_width
		self.image_height = image_height
		self.batch_size = batch_size
		self.training_data = None
		self.evaluation_data = None

		self.load_data_paths()

	def load_data_paths(self, ):
		train_path = os.path.join(self.data_path, 'images_background')
		evaluation_path = os.path.join(self.data_path, 'images_evaluation')

		for alphabet in os.listdir(train_path):
			print('[Training Data Paths] loading paths : {}'.format(alphabet))
			alphabet_path = os.path.join(train_path, alphabet)
			alphabet_dictionary = {}
			for character in os.listdir(alphabet_path):
				character_paths = os.path.join(alphabet_path, character)
				alphabet_dictionary[character] = os.listdir(character_paths)
			self.train_paths_dictionary[alphabet] = alphabet_dictionary
		
		for alphabet in os.listdir(evaluation_path):
			print('[Evaluation Data Paths] loading paths : {}'.format(alphabet))
			alphabet_path = os.path.join(evaluation_path, alphabet)
			alphabet_dictionary = {}
			for character in os.listdir(alphabet_path):
				character_paths = os.path.join(alphabet_path, character)
				alphabet_dictionary[character] = os.listdir(character_paths)
			self.evaluation_paths_dictionary[alphabet] = alphabet_dictionary

	def load_images(self, train=True):
		alphabets_to_categories = {}
		X = []
		y = []
		# categories_to_alphabet_character = {}
		current_class = 0
		paths_dictionary = self.train_paths_dictionary if train is True else self.evaluation_paths_dictionary
		data_path = os.path.join(self.data_path, 'images_background') if train is True \
			else os.path.join(self.data_path, 'images_evaluation')
		for alphabet in paths_dictionary:
			print('[{} Data] Loading Images : {}'.format('Training' if train is True else 'Test', alphabet))
			alphabets_to_categories[alphabet] = [current_class, None]
			for character in paths_dictionary[alphabet]:
				# categories_to_alphabet_character[current_class] = [alphabet, character]
				category_images = []
				for file in os.listdir(os.path.join(data_path, alphabet, character)):
					image = imageio.imread(os.path.join(data_path, alphabet, character, file))
					category_images.append(image)
					y.append(current_class)
				try:
					X.append(np.stack(category_images))
				except ValueError as e:
					print(e)
					print("error : catergory images : {}".format(category_images))
				current_class += 1
			alphabets_to_categories[alphabet][1] = current_class - 1
		y = np.vstack(y)
		X = np.stack(X)
		return X, y, alphabets_to_categories

	def load_data(self, ):
		self.load_data_paths()
		self.training_data = self.load_images()
		self.evaluation_data = self.load_images(train=False)

	def store_data(self, path=None):
		if path is None: path = self.data_path
		with open(os.path.join(path, 'training_data.pkl'), 'wb') as writeFile:
			pickle.dump(self.training_data, writeFile)
		print('Training Data written : {}'.format(os.path.join(path, 'training_data.pkl')))
		with open(os.path.join(path, 'evaluation_data.pkl'), 'wb') as writeFile:
			pickle.dump(self.evaluation_data, writeFile)
		print('Evaluation Data written : {}'.format(os.path.join(path, 'evaluation_data.pkl')))


if __name__ == '__main__':
	data_path = './../fellowship.ai/omniglot/python'
	loader = DataLoader(data_path)
	loader.load_data()
	train_X, train_y, train_alphabets_to_start_end = loader.training_data
	test_X, test_y, test_alphabets_to_start_end = loader.evaluation_data
	loader.store_data()
	from PIL import Image
	print('X.shape() =', train_X.shape)
	print('y.shape() =', train_y.shape)
	print(train_y[:50])
	print(train_alphabets_to_start_end)
	img = Image.fromarray(train_X[0,0,:,:])


	# from batch_generator import BatchManager
	# training_batch_manager = BatchManager(loader.training_data)
	# evaluation_batch_manager = BatchManager(loader.evaluation_data)
	# for (pairs, targets) in training_batch_manager.generator(5):
	# 	print(pairs.shape)
	# 	print(targets.shape)
	# for (pair, targets) in evaluation_batch_manager.generator(5):
	# 	print(pairs.shape)
	# 	print(targets.shape)