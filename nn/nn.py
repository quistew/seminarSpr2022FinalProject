from sklearn.neural_network import MLPClassifier
import numpy as np
from util.Plotter import Plotter
from sklearn.metrics import confusion_matrix


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


def main():
    X_train = np.loadtxt("../data/training.txt")
    X_test = np.loadtxt("../data/test.txt")

    classifier = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1, max_iter=300, activation="relu")     # changed from lbfgs
    classifier.fit(X_train[:,0:3], X_train[:,3])

    predictions = classifier.predict(X_test[:,0:3])

    # compare = []
    # for i in range(0, X_test.shape[0]):
    #     compare.append([predictions[i], X_test[i][3]])
    #
    # print(np.array(compare))

    cm = confusion_matrix(X_test[:,3], predictions)
    print("Accuracy of MLPClassifier:", accuracy(cm))

    # plotter = Plotter()
    # plotter.plot3D(X_train)
    # plotter.show()


main()
