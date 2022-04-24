from sklearn.neural_network import MLPClassifier
import numpy as np

X = np.loadtxt("../data/swissroll.txt")
labels = np.loadtxt("../data/preswissroll_labels.txt")

classifier = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
classifier.fit(X, labels)

