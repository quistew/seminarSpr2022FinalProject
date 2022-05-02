import numpy as np
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier

def main():
    X_train = np.loadtxt("../data/training.txt")
    X_test = np.loadtxt("../data/test.txt")

    embedding = Isomap(n_neighbors=5, n_components=2)
    X_train_transformed = embedding.fit_transform(X_train[:, 0:3])
    X_test_transformed = embedding.fit_transform(X_test[:, 0:3])
    print(X_train_transformed.shape)




main()
