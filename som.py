import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tqdm import tqdm
from collections import Counter


@nb.njit(parallel=True, cache=True)
def update_weights(W, x, influence, lr):
	for i in nb.prange(len(W)):
		W[i] = W[i] + lr * influence[i] * (x - W[i])

	return W


@nb.njit(parallel=True, cache=True)
def euclidean_dists_m_v(mat, vec):
	N = len(mat)
	dists = np.empty(N)

	for i in nb.prange(N):
		dists[i] = np.sum((mat[i] - vec) ** 2) ** .5

	return dists


@nb.njit(parallel=True, cache=True)
def euclidean_dists_m_m(mat1, mat2):
	M = len(mat1)
	N = len(mat2)
	dists = np.empty((M, N))

	for i in nb.prange(M):
		for j in nb.prange(N):
			dists[i, j] = np.sum((mat1[i] - mat2[j]) ** 2) ** .5

	return dists


@nb.njit(cache=True)
def are_neighbors(coords1, coords2):
	return (coords1[0] - 1 <= coords2[0] <= coords1[0] + 1) and (coords1[1] - 1 <= coords2[1] <= coords1[1] + 1)


class SOM:
	def __init__(self, X, m, n, n_iterations=25, alpha=0.3, sigma=None):
		np.random.seed(42)

		self.X = X
		self.m = m  # Rows
		self.n = n  # Columns
		self.dim = X.shape[-1]  # Dimensionality of the input data
		self.n_iterations = n_iterations
		self.alpha = alpha  # Initial learning rate
		self.sigma = sigma if sigma else max(m, n) / 2  # Initial neighborhood radius

		self.q_err = []
		self.t_err = []

		# Pre-calculate all sigmas and learning rates
		self.sigmas = [sigma * np.exp(-t / (self.n_iterations / 2)) for t in range(self.n_iterations)]
		self.lrs = [0.005 + (alpha - 0.005) * np.exp(-(5 / self.n_iterations) * t) for t in range(self.n_iterations)]

		# Create a grid of neuron indices
		self.grid = np.array([[i, j] for i in range(m) for j in range(n)])

		# Pre-calculate influences for all cells and iterations
		dists = euclidean_dists_m_m(self.grid, self.grid)
		self.influences = np.array([np.exp(-dists ** 2 / (2 * sig ** 2)) for sig in self.sigmas])

		# Initialize the weights
		self.weights = self.initialize_weights_mean_std(X)
		# self.weights = self.initialize_weights_subset(X)

	def initialize_weights_mean_std(self, X):
		mean = np.mean(X, axis=0).reshape(1, -1)
		std = np.std(X, axis=0)
		rnd = np.random.uniform(low=-0.2, high=0.2, size=(self.m * self.n, self.dim))
		return mean + rnd * std

	def initialize_weights_subset(self, X):
		"""Initialize the weights based on a subset of the data."""
		# Shuffle the data and select a subset for initialization
		X_subset = shuffle(X)[:self.m * self.n]
		return X_subset

	def train(self, log_err=True):
		for t in tqdm(range(self.n_iterations)):
			lr = self.lrs[t]

			for x in self.X:
				bmu_idx = self.find_bmu(x)
				influence = self.influences[t, bmu_idx]
				self.weights = update_weights(self.weights, x, influence, lr)

			if log_err:
				q_err, t_err = self.calculate_errors()
				self.q_err.append(q_err)
				self.t_err.append(t_err)

	def find_bmu(self, x, *, two=False):
		# Calculate the Euclidean distance between the input and all the neurons
		distances = euclidean_dists_m_v(self.weights, x)

		return np.argsort(distances)[:2] if two else np.argmin(distances)

	def calculate_errors(self):
		q_err, t_err = 0, 0
		for x in self.X:
			bmu_indices = self.find_bmu(x, two=True)
			q_err += np.linalg.norm(x - self.weights[bmu_indices[0]])

			bmu_coords = self.grid[bmu_indices[0]]
			second_bmu_coords = self.grid[bmu_indices[1]]
			if not are_neighbors(bmu_coords, second_bmu_coords):
				t_err += 1

		return q_err / len(self.X), t_err / len(self.X)

	def visualize_weights(self, y, *, k=100, figsize=(12, 12)):
		print(f"Final Quantization Error: {self.q_err[-1]}")
		print(f"Final Topographic Error: {self.t_err[-1]}")

		# Find the label and confidence for each neuron
		labels = np.zeros(self.m * self.n)
		confidences = np.zeros(self.m * self.n)

		for i in range(self.m * self.n):
			# Find the samples that are closest to this neuron
			distances = np.linalg.norm(self.X - self.weights[i], axis=1)
			closest_samples_indices = np.argsort(distances)[:k]  # Take k closest samples
			closest_labels = y[closest_samples_indices]

			# Determine the most common label and its confidence
			label_counter = Counter(closest_labels)
			most_common_label, most_common_count = label_counter.most_common(1)[0]

			labels[i] = most_common_label
			confidences[i] = most_common_count / k

		# Visualize the neurons with the digits
		fig, axes = plt.subplots(self.m, self.n, figsize=figsize)
		for i, ax in enumerate(axes.flat):
			neuron_weights = self.weights[i].reshape(28, 28)
			ax.imshow(neuron_weights, cmap='gray')
			ax.set_title(f'{int(labels[i])} ({confidences[i]:.2f})')
			ax.axis('off')

		plt.tight_layout()
		plt.show()

	def get_errors(self, *, prnt=False):
		if prnt:
			print(f"Final Quantization Error: {self.q_err[-1]}")
			print(f"Final Topographic Error: {self.t_err[-1]}")

		return {"q": np.array(self.q_err),
		        "t": np.array(self.t_err)}
