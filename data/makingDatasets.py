import numpy as np
from PreProcessor import PreProcessor

X_orig = np.loadtxt("swissroll.txt")
labels = np.loadtxt("preswissroll_labels.txt")

X_full_scaled = PreProcessor(X_orig).rescale()
X_full = np.concatenate((X_full_scaled,np.array([labels]).transpose()),1)
np.random.shuffle(X_full)

X_train = X_full[0:1500,:]
X_test = X_full[1500:,:]

np.savetxt("training.txt", X_train)
np.savetxt("test.txt", X_test)
