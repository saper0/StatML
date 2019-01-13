#######################################################################
# Author: Lukas Gosch
# Date: 04.11.2018
# Description: This script implements the stochastic subgradient method
#			   for the wine dataset but has been used for experimentation
#			   with different stepsizes and to plot the error rates against
#			   number of steps.
#######################################################################

import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def trainMultiSVM(X, Y, T, lmbda=1):
	""" Train a multiclass SVM using the stochastic subgradient method
		and one-versus-all approach. 

		X ... training feature vectors
		Y ... true labels element of [1, K]
		T ... iterations
		lmbda ... coefficient controlling how strong a wrong
				   classification should be weighted
				   (small lmbda => high weight)
	"""
	labels = [int(k) for k in set(Y)]
	# one - versus - all
	h_l = []
	eval_l = []
	for label in labels:
		# construct binary labeling
		Y_bnry = [1 if y == label else -1 for y in Y]
		# train binary SVM
		w, b, w_l, b_l = trainSVM(X, Y_bnry, T, lmbda)
		# add SVM to binary predictor list
		h_l.append((w,b))
		# store for later evaluation of learning
		eval_l.append((w_l,b_l))
	return h_l, eval_l

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
	w_l = []
	b_l = []
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
		# store w, b for later evaluation of learning process
		w_l.append(1/t * w_avg)
		b_l.append(1/t * b_avg)
		#w_l.append(w)
		#b_l.append(b)
	return 1/T * w_avg, 1/T * b_avg, w_l, b_l

def trainSVMConstStep(X, Y, T, lmbda=1):
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
	w_l = []
	b_l = []
	for t in range(1, T+1):
		# w_t
		w = 1 / lmbda * theta 
		# b_t
		b = 1 / lmbda * theta_0
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
		# store w, b for later evaluation of learning process
		w_l.append(1/t * w_avg)
		b_l.append(1/t * b_avg)
		#w_l.append(w)
		#b_l.append(b)
	return 1/T * w_avg, 1/T * b_avg, w_l, b_l

def trainSVMSqrtStep(X, Y, T, lmbda=1):
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
	w_l = []
	b_l = []
	for t in range(1, T+1):
		# w_t
		w = 1 / (lmbda * math.sqrt(t)) * theta 
		# b_t
		b = 1 / (lmbda * math.sqrt(t)) * theta_0
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
		# store w, b for later evaluation of learning process
		w_l.append(1/t * w_avg)
		b_l.append(1/t * b_avg)
		#w_l.append(w)
		#b_l.append(b)
	return 1/T * w_avg, 1/T * b_avg, w_l, b_l

def trainSVMPowerStep(X, Y, T, lmbda=1):
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
	w_l = []
	b_l = []
	for t in range(1, T+1):
		# w_t
		w = 1 / (lmbda * t*t) * theta 
		# b_t
		b = 1 / (lmbda * t*t) * theta_0
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
		# store w, b for later evaluation of learning process
		w_l.append(1/t * w_avg)
		b_l.append(1/t * b_avg)
		#w_l.append(w)
		#b_l.append(b)
	return 1/T * w_avg, 1/T * b_avg, w_l, b_l

def applyMultiSVM(x, h_l):
	""" Apply a trained SVM for multiclass classification.
		
		x ... datapoint to predict
		h_l ... list of binary predictors
	"""
	# find predictor with highest confidence
	max_confidence = np.inner(h_l[0][0], x) + h_l[0][1]
	prediction = 0
	for k in range(1, len(h_l)):
		w, b = h_l[k]
		confidence = np.inner(w, x) + b
		if confidence > max_confidence:
			max_confidence = confidence
			prediction = k
	return prediction + 1

def predictionError(X, Y, h_l):
	""" Return percentage of wrong classified datapoints. 
		
		X ... Test data, Y ... true labels
		w ... weight vector SVM, b .. bias term SVM
	"""
	pred = []
	for x in X:
		pred.append(applyMultiSVM(x, h_l))

	pred = np.array(pred)

	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y)
	return cnt_wrng / np.size(pred)

def calcObjective(lmbda, w, b, X, Y_bnry):
	""" Calculate objective value of a given binary SVM. """
	avg_err = 0
	for i in range(0, X.shape[0]):
		err = 1 - Y_bnry[i] * (np.inner(w, X[i,]) + b)
		if err > 0:
			avg_err = avg_err + err
	avg_err = 1/X.shape[0] * avg_err

	return lmbda / 2 * np.inner(w, w) + avg_err

def evalLearning(h_l, eval_l, lmbda, T, X, Y, X_test, Y_test):
	""" Return list of SVM's objective values and the euclidian
		distances to the optimum for each iteration and binary SVM.
	"""
	labels = [int(k) for k in set(Y)]
	obj_vals = []
	for label in labels:
		# construct binary labeling
		Y_bnry = [1 if y == label else -1 for y in Y]
		# retrieve optimal classifier
		w, b = h_l[label-1]
		# retrieve w, b during the learning process
		w_l, b_l = eval_l[label-1]
		# calculate SVM's objective value
		obj_val = []
		for i in range(0, T):
			obj_val.append(calcObjective(lmbda, w_l[i], b_l[i], X, Y_bnry))
		obj_vals.append(obj_val)
	
	err_rates_trn = []
	err_rates_test = []
	for t in range(0, T):
		# calculate error rates for each number of steps
		svm_l = [] 
		for label in labels:
			# retrieve w, b during the learning process
			w_l, b_l = eval_l[label-1]
			svm_l.append((w_l[t], b_l[t]))
		err_rates_trn.append(predictionError(X, Y, svm_l))
		err_rates_test.append(predictionError(X_test, Y_test, svm_l))

	return obj_vals, err_rates_trn, err_rates_test

# Prepare training data
data = np.loadtxt('wine-train.txt')
X = data[:,1:]
# scale to zero mean and unit variance
X = preprocessing.scale(X)
Y = data[:,0]

# Train SVM
np.random.seed(1)
lmbda=0.27
T = 40
h_l, eval_l = trainMultiSVM(X, Y, T, lmbda)

# Apply SVM on training data
print('Application on training data:')
predEr = predictionError(X, Y, h_l)
print('Prediction Error: ', predEr)

# Prepare test data
data = np.loadtxt('wine-test.txt')
X_test = data[:,1:]
# scale to zero mean and unit variance
X_test = preprocessing.scale(X_test)
Y_test = data[:,0]

# Apply SVM on test data
print('Application on test data:')
predEr = predictionError(X_test, Y_test, h_l)
print('Prediction Error: ', predEr)

# Evaluate data for report
obj_vals, err_rates_trn, err_rates_test= evalLearning(h_l, eval_l, lmbda, T, X, Y, X_test, Y_test)

fig, (first, second) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('SVM Learning Process')
# Plot SVM's objective values
first.plot(range(1, T+1),obj_vals[0], 'o-',color='g', label='Binary SVM Label 1 vs all')
first.plot(range(1, T+1),obj_vals[1], 'o-',color='y', label='Binary SVM Label 2 vs all')
first.plot(range(1, T+1),obj_vals[2], 'o-',color='b', label='Binary SVM Label 3 vs all')
first.set_ylabel('SVM\'s objective value')
first.set_xlabel('Iterations')
first.legend()

# Plot euclidean distances to the optimum
second.plot(range(1, T+1),err_rates_trn, 'o-',color='g', label='Training Error')
second.plot(range(1, T+1),err_rates_test, 'o-',color='y', label='Test Error')
second.set_ylabel('Error Rates at stepsize t')
second.set_xlabel('Iterations')
second.legend()

plt.show()







