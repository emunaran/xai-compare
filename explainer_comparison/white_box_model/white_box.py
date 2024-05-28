import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
from sklearn.metrics import euclidean_distances


class WhiteBoxRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        # Known coefficients
        self.coefficients_ = np.array([2, 3, -1, 5, -4, 0.5, 2, -3, 0.1, 1])

    def fit(self, X, y=None):
        # No fitting necessary for whitebox model with fixed coefficients
        return self

    def predict(self, X):
        # Prediction function using known coefficients
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        y_pred = (self.coefficients_[0] * X[:, 0] +
                  self.coefficients_[1] * X[:, 1] +
                  self.coefficients_[2] * X[:, 2] +
                  self.coefficients_[3] * X[:, 3] +
                  self.coefficients_[4] * X[:, 4] +
                  self.coefficients_[5] * X[:, 5] +
                  self.coefficients_[6] * X[:, 6] +
                  self.coefficients_[7] * X[:, 7] +
                  self.coefficients_[8] * X[:, 8] +
                  self.coefficients_[9] * X[:, 9])
        return y_pred

    def __call__(self, X):
        return self.predict(X)

    def generate_synthetic_data(self, n_samples=1000, with_y = True, add_noise = False):
        # Generate the synthetic dataset
        np.random.seed(42)

        X = np.zeros((n_samples, 10))
        X[:, 0] = np.linspace(0, 10, n_samples)
        X[:, 1] = np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.sin(np.linspace(0, 4*np.pi, n_samples))**2
        X[:, 2] = np.sin(np.linspace(0, 4*np.pi, n_samples))
        X[:, 3] = np.sin(np.linspace(0, 4*np.pi, n_samples))
        X[:, 4] = np.cos(np.linspace(0, 4*np.pi, n_samples))
        X[:, 5] = np.exp(np.linspace(0, 2, n_samples))
        X[:, 6] = np.log(np.linspace(1, 11, n_samples))
        X[:, 7] = np.cos(np.linspace(0, 4*np.pi, n_samples))**2
        X[:, 8] = np.sin(np.linspace(0, 2*np.pi, n_samples))# + np.linspace(0, 10, n_samples)
        X[:, 9] = np.random.normal(0, 1, n_samples) if add_noise == True else np.linspace(0, 10, n_samples)

        df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(10)])

        if with_y:
            df['y'] = self.predict(df)

        return df

    @staticmethod
    def find_closest_sample_and_explanation(sample, X):
        # Function to find the closest sample and calculate the actual explanation
        # Find the closest sample in the dataset
        distances = euclidean_distances([sample], X)
        closest_idx = np.argmin(distances)
        closest_sample = X.iloc[closest_idx]

        return closest_sample


if __name__ == '__main__':
    model = WhiteBoxRegressor()
    df = model.generate_synthetic_data(1000, with_y=True)
    df.plot()
    plt.show()
