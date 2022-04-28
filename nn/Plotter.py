import matplotlib.pyplot as plt

class Plotter():

    def __init__(self):
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = plt.axes(projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def plot3D(self, X, color=None):
        colorDict = {1: "cornflowerblue", 2: "green", 3: "blue", 4: "purple"}
        if color == None:
            for row in X:
                self.ax.scatter3D(row[0], row[1], row[2], color=colorDict[row[-1]])
        else:
            for row in X:
                self.ax.scatter3D(row[0], row[1], row[2], color=color)

    def show(self):
        plt.show()

