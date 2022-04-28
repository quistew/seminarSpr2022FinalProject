from sklearn.neural_network import MLPClassifier
import numpy as np
# from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt
from Plotter import Plotter
from PreProcessor import PreProcessor
from Sampler import Sampler

X = np.loadtxt("../data/swissroll.txt")
labels = np.loadtxt("../data/preswissroll_labels.txt")

pp = PreProcessor(X)
X_pp = pp.rescale()

X_full = np.concatenate((X_pp, np.array([labels]).transpose()), 1)
X_train = Sampler(X_full).getRandomSample(600)


classifier = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1, max_iter=5000)     # changed from lbfgs
classifier.fit(X, labels)

newPoint1 = np.array([[-1.5, -1.5, -1.5]])
print(classifier.predict(newPoint1.reshape(1,-1)))

newPoint2 = np.array([[1, 1.5, 1]])
print(classifier.predict(newPoint2.reshape(1,-1)))

newPoint3 = np.array([[.5, 1.5, 1.5]])
print(classifier.predict(newPoint3.reshape(1,-1)))

plotter = Plotter()
plotter.plot3D(X_train)
plotter.plot3D(newPoint1, "red")
plotter.plot3D(newPoint2, "red")
plotter.plot3D(newPoint3, "red")
plotter.show()

# TODO:
# - Make a single dataset, mix classifications order (and make that a fourth column in the data)
# - Need to batch train? Watch tutorial.
