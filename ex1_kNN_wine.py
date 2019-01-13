#######################################################################
# Author: Lukas Gosch
# Date: 14.10.2018
# Description: Implementation of a k-NN algorithm for the wine dataset.
#######################################################################

import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

def kNNpredict(X, Y, K, p):
	""" Apply k-NN to predict label of p. 
		
		X ... training feature vectors
		Y ... label vector
		K ... k in k-NN (if k is even and a tie happens, returns 1)
		x ... point to predict label for
	"""
	# Calculate distance to every training point
	distances = []
	for i in range(0, X.shape[0]):
		x = X[i]
		distances.append((np.linalg.norm(p-x), i))
	# Sort distances using timsort. Timsort is stable, this is an 
	# important property, as this means if two training samples are
	# the same distance to p but we can only choose one in our k-NN
	# list, the training sample which distance got calculated first
	# will be used.
	distances.sort(key=lambda tup: tup[0]) 
	# Count most common label, if two elements have equal (most common)
	# count, the order is arbitrary.
	y = []
	for k in range(0, K):
		y.append(Y[distances[k][1]])
	counter = collections.Counter(y).most_common(1)
	# Return majority vote decision
	return counter[0][0]
	

def predictionError(X, Y, X_test, Y_test, k):
	""" Return percentage of wronge classified datapoints when applying
		k-NN trained on X to the dataset X_test.

		X ... training feature vectors
		Y ... training labels
		X_test ... test set
		k ... k in k-NN
	"""
	# Predict labels for X_test
	pred = []
	for x in X_test:
		pred.append(kNNpredict(X, Y, k, x))
	# Evaluate prediction
	pred = np.array(pred)
	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y_test)
	return cnt_wrng / np.size(pred)

# Prepare training data
data = np.loadtxt('wine-train.txt')
X = scale(data[:,1:])
Y = data[:,0]
K = range(1, 45, 2)

# Apply k-NN on training data
print('k-NN on training data:')
complexity_vs_train = []
for k in K:
	predEr = predictionError(X, Y, X, Y, k)
	complexity_vs_train.append(predEr)
	print('Prediction Error: ', predEr, ' k: ', k)

# Prepare test data
data = np.loadtxt('wine-test.txt')
X_test = scale(data[:,1:])
Y_test = data[:,0]

# Apply k-NN on test data
print('Application on test data:')
complexity_vs_test = []
for k in K:
	predEr = predictionError(X, Y, X_test, Y_test, k)
	complexity_vs_test.append(predEr)
	print('Prediction Error: ', predEr, ' k: ', k)

# Plot complexity vs training/test error
plt.plot(K, complexity_vs_train, 'o-', color='g', label='Training error')
plt.plot(K, complexity_vs_test, 'o-', color='y', label='Test error')
plt.title('Complexity vs Error')
plt.ylabel('Prediction Error')
plt.xlabel('k in k-NN')
plt.xticks(K)
plt.legend()
plt.show()
