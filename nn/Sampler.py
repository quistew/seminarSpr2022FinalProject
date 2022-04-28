import numpy as np


class Sampler:

    def __init__(self, data):
        self.data = data

    def getRandomSample(self, size):
        if self.data.shape[0] < size:
            print("Size was bigger than size of dataset:", size, ">", self.data.shape[0])
            return None
        else:
            id_rand = np.random.randint(0, self.data.shape[0]-1, size=size)
            return self.data[id_rand, :]



# def main():
#     X = np.loadtxt("../data/swissroll.txt")
#     y = np.loadtxt("../data/preswissroll_labels.txt")
#     combined = np.concatenate((X,np.array([y]).transpose()),1)
#     sampled = Sampler(combined).getRandomSample(50)
#     print(sampled)
#     print(sampled[:,0:3])
#
# main()