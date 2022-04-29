from sklearn.neural_network import MLPClassifier
import numpy as np
from util.Plotter import Plotter

X_train = np.loadtxt("../data/training.txt")
X_test = np.loadtxt("../data/test.txt")

classifier = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1, max_iter=5000, activation="relu")     # changed from lbfgs
classifier.fit(X_train[:,0:3], X_train[:,3])

plotter = Plotter()
plotter.plot3D(X_train)
# plotter.plot3D(newPoint1, "red")
# plotter.plot3D(newPoint2, "red")
# plotter.plot3D(newPoint3, "red")
plotter.show()


