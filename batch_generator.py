import numpy as np

class BatchManager:
	def __init__(self, loaded_data):
		(self.X, self.y, self.categories) = loaded_data
		
	def getNewBatch(self, batch_size=1000):
		classes, items_per_class, width, height = self.X.shape
		chosen_categories = np.randome.choice(classes, size=(batch_size, ), replace=False)

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
			pairs, targets = self.getNewBatch(batch_size)
			yield (pairs, targets)