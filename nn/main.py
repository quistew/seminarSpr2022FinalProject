from sklearn.neural_network import MLPClassifier
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

X = np.loadtxt("../data/swissroll.txt")
labels = np.loadtxt("../data/preswissroll_labels.txt")

classifier = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)     # changed from lbfgs
classifier.fit(X, labels)
print(classifier.predict(np.array([-7.5, -7.5, 7.5]).reshape(1,-1)))

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.scatter3D(X[:,0], X[:,1], X[:,2])
plt.show()