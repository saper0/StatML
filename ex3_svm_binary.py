#######################################################################
# Author: Lukas Gosch
# Date: 28.10.2018
# Description: Implementation of a linear SVM for the binary dataset 
#			   Optimization is done using a stochastic subgradient 
#			   method. The implementation follows roughly the pseudocode
#			   in chapter 15.5 S213 in "Understanding Machine Learning: 
#			   From Theory to Algorithms" (Shwartz) titled SGD for 
#			   Solving Soft-SVM.
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

def trainSVM(X, Y, T, lmbda=1):
	""" Train a linear SVM using the stochastic subgradient method. 

		X ... training feature vectors
		Y ... true labels element of {-1, 1}
		T ... iterations
		lmbda ... coefficient controlling how strong a wrong
				   classification should be weighted
				   (small lmbda => high weight)
	"""
	N = X.shape[0]
	theta = np.zeros((X.shape[1],))
	theta_0 = 0
	w_avg = np.zeros((X.shape[1],))
	b_avg = 0
	for t in range(1, T+1):
		# w_t
		w = 1 / (lmbda * t) * theta 
		# b_t
		b = 1 / (lmbda * t) * theta_0
		# choose (uniformly distributed) random data point
		i = np.random.randint(0, N)
		x_i = X[i,:]
		y_i = Y[i]
		# update using subgradient of objective at w_t
		if y_i * np.inner(w, x_i) + b < 1:
			theta = theta + y_i * x_i
			theta_0 = theta_0 + y_i
		w_avg = w_avg + w
		b_avg = b_avg + b
	return 1/T * w_avg, 1/T * b_avg

def applySVM(x, w, b):
	""" Apply a trained SVM for binary classification of datapoint x. """
	if np.inner(w, x) + b > 0:
		return 1
	return -1

def predictionError(X, Y, w, b):
	""" Return percentage of wrong classified datapoints. 
		
		X ... Test data, Y ... true labels
		w ... weight vector SVM, b .. bias term SVM
	"""
	pred = []
	for x in X:
		pred.append(applySVM(x, w, b))

	pred = np.array(pred)

	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y)
	return cnt_wrng / np.size(pred)

# Prepare training data
X, Y = genTrainingData(swapped = False)
X = np.array(X)
Y = np.array(Y)

X_1, Y_1 = genTrainingData(swapped = True)
X_1 = np.array(X_1)
Y_1 = np.array(Y_1)

# Train SVM
np.random.seed(28)
# small lambda for linear separable case
w, b = trainSVM(X, Y, T=100, lmbda=1e-6)
w_1, b_1 = trainSVM(X_1, Y_1, T=100, lmbda=0.1)

# Apply SVM on training data
print('Application on training data with y9=1:')
predEr = predictionError(X, Y, w, b)
print('Prediction Error: ', predEr)
print('Application on training data with y9=-1:')
predEr = predictionError(X_1, Y_1, w_1, b_1)
print('Prediction Error: ', predEr)

# Prepare test data
X_test, Y_test = genTestData()
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Apply SVM on test data
print('Application on test data with SVM trained on y9=1:')
predEr = predictionError(X_test, Y_test, w, b)
print('Prediction Error: ', predEr)
print('Application on test data with SVM trained on y9=-1:')
predEr = predictionError(X_test, Y_test, w_1, b_1)
print('Prediction Error: ', predEr)

# Calculate averages over multiple runs
runs = 1000
avg_train = 0
avg_train_1 = 0
avg_test = 0
avg_test_1 = 0
for run in range(0, runs):
	np.random.seed(run)
	# small lambda for linear separable case
	w, b = trainSVM(X, Y, T=100, lmbda=1e-6)
	w_1, b_1 = trainSVM(X_1, Y_1, T=100, lmbda=0.1)

	# Apply SVM on training data
	avg_train = avg_train + predictionError(X, Y, w, b)
	avg_train_1 = avg_train_1 + predictionError(X_1, Y_1, w_1, b_1)

	# Apply SVM on test data
	avg_test = avg_test + predictionError(X_test, Y_test, w, b)
	avg_test_1 = avg_test_1 + predictionError(X_test, Y_test, w_1, b_1)

avg_train = 1/runs * avg_train
avg_train_1 = 1/runs * avg_train_1
avg_test = 1/runs * avg_test
avg_test_1 = 1/runs * avg_test_1

print("Average prediction erros over ", runs, " runs:")
print("y = 1: Train error ", avg_train)
print("y = 1: Test error ", avg_test)
print("y = -1: Train error ", avg_train_1)
print("y = -1: Test error ", avg_test_1)

