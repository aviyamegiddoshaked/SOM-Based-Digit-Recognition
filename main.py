from som import SOM

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main():
	# Load MNIST data
	X = pd.read_csv('digits_test.csv', names=range(784)).values
	y = pd.read_csv('digits_test_key.csv', names=[0]).values.reshape(-1)
	dir_name = 'results-new'

	for dim in [10]: #, 12, 15, 20]:
		for s_lr in [0.3]: #[1., 0.7, 0.3, 0.1, 0.05]:
			for s_rad in [3]: #[2, 3, 4, 5]:
				for n_it in [30]:
					# Scale the data
					scaler = MinMaxScaler()
					X_scaled = scaler.fit_transform(X.copy())

					# Initialize SOM
					som = SOM(X_scaled, m=dim, n=dim, n_iterations=n_it, alpha=s_lr, sigma=s_rad)
					cfg = f"dim-{som.m}x{som.n}_lr-{som.alpha}_sig-{som.sigma}_it-{som.n_iterations}"
					print(cfg)
					som.train()

					# Visualize the SOM with neurons and digits
					som.visualize_weights(y, figsize=(12, 12))

					# name = f"{cfg}.csv"
					# print(f"Done {name}")

					# err = som.get_errors()
					# print()

					# df = pd.DataFrame(data=err)
					# df.to_csv(os.path.join(dir_name, name), index=False)


if __name__ == '__main__':
	main()
