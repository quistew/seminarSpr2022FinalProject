from sklearn.neural_network import MLPClassifier
import numpy as np
# from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt
from Plotter import Plotter
from PreProcessor import PreProcessor

X = np.loadtxt("../data/swissroll.txt")
labels = np.loadtxt("../data/preswissroll_labels.txt")

pp = PreProcessor(X)
X_train = pp.rescale()

classifier = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1, max_iter=5000)     # changed from lbfgs
classifier.fit(X, labels)

newPoint1 = np.array([[-1.5, -1.5, -1.5]])
print(classifier.predict(newPoint1.reshape(1,-1)))

newPoint2 = np.array([[1, 1.5, 1]])
print(classifier.predict(newPoint2.reshape(1,-1)))

plotter = Plotter()
plotter.plot(X_train, "blue")
plotter.plot(newPoint1, "red")
plotter.plot(newPoint2, "red")
plotter.show()
