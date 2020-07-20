import numpy as np
# from sklearn.utils import shuffle

class BatchManager:
	def __init__(self, loaded_data):
		self.width = 105
		self.height = 105
		(self.X, self.y, self.categories) = loaded_data
		
	def get_batch(self, batch_size=30):
		classes, items_per_class, width, height = self.X.shape
		chosen_categories = np.random.choice(classes, size=(batch_size, ), replace=False)

		pairs = [np.zeros((batch_size, width, height, 1)) for _ in range(2)]

		targets = np.zeros((batch_size, ))
		targets[batch_size//2:] = 1

		count = 0
		for category in chosen_categories:
			chosen_item = np.random.randint(0, items_per_class)
			pairs[0][count, :, :, :] = self.X[category, chosen_item].reshape(width, height, 1)
			if count < batch_size // 2:
				category = (category + np.random.randint(1, classes)) % classes
				chosen_item = np.random.randint(0, items_per_class)
			else:
				chosen_item = (chosen_item + np.ranint(1, items_per_class)) % items_per_class
			pairs[1][count, :, :, :] = self.X[category, chosen_item].reshape(width, height, 1)
		return pairs, targets

	def generator(self, batch_size):
		while True:
			pairs, targets = self.get_batch(batch_size)
			yield (pairs, targets)

	def get_test_batch(self, batch_size):
		indices = np.random.randint(low=0, high=20, size=(batch_size, ))
		classes, items_per_class, width, height = self.X.shape
		chosen_categories = np.random.choice(range(classes), size=(batch_size, ), replace=False)
		true_category = chosen_categories[0]
		item1, item2 = np.random.choice(items_per_class, size=(2, ), replace=False)
		# create N copies for the true category item for evaluation in 'batch_size' way
		test_image = np.asarray([self.X[true_category, item1, : , :]] * batch_size).reshape( \
			batch_size, width, height, 1)
		support_images = self.X[chosen_categories, indices, :, :]
		support_images[0, :, :] = self.X[true_category, item2]
		support_images = support_images.reshape(batch_size, self.width, self.height, 1)
		targets = np.zeros((batch_size, ))
		targets[0] = 1
		# targets, test_image, support_images = shuffle(targets, test_image, support_images)
		pairs = [test_image, support_images]
		return pairs, targets

	# def get_test_batch(self, batch_size, same_alphabets=True, alphabets=None):
	# 	classes, items_per_class, width, height = self.X.shape
	# 	if same_alphabets is not None:
	# 		if self.categories[alphabets][1] - self.categories[alphabets][0] < batch_size:
	# 			raise ValueError('Alphabets: {} have doesn\'t have enough alphabet.\n \
	# 				Expected: {}, Actual number of alphabet : {}'.format(\
	# 				alphabets, batch_size, self.categories[alphabets][1]-self.categories[alphabets][0]))
	# 		low, high = self.categories[alphabets]
	# 		characters = np.random.choice(range(low, high), size=(batch_size, ), replaec=False)
	# 	elif same_alphabets is True:
	# 		count = 0
	# 		while True:
	# 			if count > classes:
	# 				raise ValueError('No set of alphabets has enough characters to make a batch of size : {}'.format(batch_size	))
	# 			alphabets = np.random.choice(list(self.categories.keys()), replace=False)
	# 			low, high = self.categories[alphabets]
	# 			if high-low >= batch_size: break
	# 		characters = np.random.choice(range(low, high), size=(batch_size, ), replace=False)
	# 	else: