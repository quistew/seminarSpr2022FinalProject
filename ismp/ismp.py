import numpy as np
from sklearn.manifold import Isomap

from sklearn.datasets import load_digits

def main():
    X_train = np.loadtxt("../data/training.txt")
    # X_test = np.loadtxt("../data/test.txt")
    # X_train, _ = load_digits(return_X_y=True)

    embedding = Isomap(n_neighbors=5, n_components=2)
    X_transformed = embedding.fit_transform(X_train[:,0:3])
    print(X_transformed.shape)


main()
