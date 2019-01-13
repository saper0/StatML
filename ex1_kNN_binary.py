#######################################################################
# Author: Lukas Gosch
# Date: 14.10.2018
# Description: Implementation of a k-NN algorithm.
#######################################################################

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

def genTrainingData(swapped = False):
	""" Generate and return test data 

		Swap label of 9th training point if swapped is true. 
	"""

	X = []
	Y = []

	X.append([0, 0, 0])
	X.append([0, 0, 0])
	X.append([0, 1, 0])
	X.append([0, 1, 0])
	X.append([0, 1, 1])
	X.append([1, 0, 1])
	X.append([1, 0, 0])
	X.append([1, 0, 1])
	X.append([1, 1, 0])

	Y.append(-1)
	Y.append(-1)
	Y.append(1)
	Y.append(1)
	Y.append(1)
	Y.append(1)
	Y.append(1)
	Y.append(1)
	if not swapped:
		Y.append(1)
	else:
		Y.append(-1)

	return X, Y

def genTestData():
	""" Generate and return test data. """

	X = []
	Y = []

	X.append([1, 1, 1])
	X.append([0, 0, 1])
	X.append([0, 1, 0])
	X.append([0, 1, 1])

	Y.append(1)
	Y.append(-1)
	Y.append(-1)
	Y.append(1)

	return X, Y


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
	# Add labels of K nearest training data points
	y = 0
	for k in range(0, K):
		y = y + Y[distances[k][1]]
	# Return sign of y or 1 if y is zero
	if y < 0:
		return -1
	return 1

def main(argv):
	parser = argparse.ArgumentParser(description='Call with --swapped if 9th data point should have label -1 instead of +1.')
	# Choose if 9th trianing point should swap label
	parser.add_argument('--swapped', action='store_true', default=False)
	args = parser.parse_args(argv[1:])

	# Prepare training data
	X, Y = genTrainingData(args.swapped)
	X = np.array(X)
	Y = np.array(Y)
	features = range(0, len(X[0]))
	
	# Apply k-NN on training data
	print('k-NN on training data:')
	complexity_vs_train = []
	for k in range(1, 10, 2):
		predEr = predictionError(X, Y, X, Y, k)
		complexity_vs_train.append(predEr)
		print('Prediction Error: ', predEr, ' k: ', k)
	
	# Prepare test data
	X_test, Y_test = genTestData()
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	# Apply k-NN on test data
	print('Application on test data:')
	complexity_vs_test = []
	for k in range(1, 10, 2):
		predEr = predictionError(X, Y, X_test, Y_test, k)
		complexity_vs_test.append(predEr)
		print('Prediction Error: ', predEr, ' k: ', k)

	# Plot complexity vs training/test error
	plt.plot(range(1, 10, 2), complexity_vs_train, 'o-', color='g', label='Training error')
	plt.plot(range(1, 10, 2), complexity_vs_test, 'o-', color='y', label='Test error')
	plt.title('Complexity vs Error')
	plt.ylabel('Prediction Error')
	plt.xlabel('k in k-NN')
	plt.xticks(range(9, 0, -2))
	plt.legend()
	plt.show()
	

if __name__ == "__main__":
	main(sys.argv)