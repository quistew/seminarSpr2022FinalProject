import numpy as np


class Sampler:

    def __init__(self, data):
        self.data = data
        self.id_rand = None
        self.id_randNot = None

    def getRandomSample(self, size):
        if self.data.shape[0] < size:
            print("Size was bigger than size of dataset:", size, ">", self.data.shape[0])
            return None
        else:
            self.id_rand = np.random.randint(0, self.data.shape[0]-1, size=size)

            id_randNot_ls = []
            for i in range(0,size-1):
                if i not in self.id_rand:
                    id_randNot_ls.append(i)

            self.id_randNot = np.array(id_randNot_ls)

            print(self.id_rand)
            print(self.id_randNot)

            return (self.data[self.id_rand, :], self.data[self.id_randNot, :])



# def main():
#     # X = np.loadtxt("../data/swissroll.txt")
#     # y = np.loadtxt("../data/preswissroll_labels.txt")
#     X1 = np.random.randint(0, 100, size=10,)
#     X2 = np.random.randint(0,100,size=10)
#     X = np.vstack((X1, X2)).transpose()
#     print(X)
#     # combined = np.concatenate((X,np.array([y]).transpose()),1)
#     sampled, unsampled = Sampler(X).getRandomSample(5)
#     print(sampled)
#     print(unsampled)
#     # print(sampled[:,0:3])
#
# main()