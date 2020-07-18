import os
import numpy as np
from PIL import IMAGE as image
import random
import imageio

class DatasetLoader:
	def __init__(self, dataset_path, image_width=100, image_height=100, batch_size=100):
		self.dataset_path = dataset_path
		self.train_paths_dictionary = {}
		self.evaluation_paths_dictionary = {}
		self.image_width = image_width
		self.image_height = image_height
		self.batch_size = batch_size

		self.load_data_paths()

	def load_data_paths(self, ):
		train_path = os.path.join(self.dataset_path, 'images_background')
		validation_path = os.path.join(self.dataset_path, 'images_evaluation')

		for alphabets in os.listdir(train_path):
			alphabets_path = os.path.join(train_path, alphabets)
			alphabets_dictionary = {}
			for character in os.listdir(alphabets_path):
				characters_path = os.path.join(alphabets_path, character)
				alphabets_dictionary[character] = os.listdir(characters_path)
			self.train_paths_dictionary[alphabets] = alphabets_dictionary
		print('Paths for training data read successfully :)')
		for alphabets in os.listdir(validation_path):
			alphabets_path = os.path.join(train_path, alphabets)
			alphabets_dictionary = {}
			for character in os.listdir(alphabets_path):
				characters_path = os.path.join(alphabets_path, character)
				alphabets_dictionary[character] = os.listdir(characters_path)
			self.evaluation_paths_dictionary[alphabets] = alphabets_dictionary
		print('Paths for validation data read successfully :)')

	def path_list_to_iamges(self):
