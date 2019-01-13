#######################################################################
# Author: Lukas Gosch
# Date: 20.10.2018
# Description: Implementation of a Gaussian Naive Bayes for the 
#			   wine dataset.
#######################################################################

import sys
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import math

def gaussPdf(x, mean, var):
	""" Return probability density of x ~ N(mean, var). """
	pi = 3.1415926
	denom = (2*pi*var)**.5
	num = math.exp(-(x-mean)*(x-mean)/(2*var))
	return num/denom

def trainNaiveBayes(X, Y):
	""" Train a gaussian naive bayes classifier. 

		X ... training feature vectors
		Y ... true labels element of [1, K]
	"""
	# Calculate Prior p(y)
	N = Y.size
	counter = collections.Counter(Y)
	keys = [int(i) for i in list(counter.keys())]
	keys.sort()
	K = len(keys)
	p_y = np.zeros((K, 1))
	for key in keys:
		p_y[key-1] = counter[key] / N

	# Calculate sample mean for each class and feature
	D = len(X[0])
	Mean = np.zeros((K, D))
	for i in range(0, N):
		for k in keys:
			if Y[i] == k:
				for d in range(0, D):
					Mean[k-1,d] = Mean[k-1,d] + X[i,d]
				break
	for k in keys:
		for d in range(0, D):
			Mean[k-1,d] = Mean[k-1,d] / counter[k]

	# Calculate (biased) sample variance for each class and feature
	Var = np.zeros((K, D))
	for i in range(0, N):
		for k in keys:
			if Y[i] == k:
				for d in range(0, D):
					Var[k-1,d] = Var[k-1,d] + (X[i,d] - Mean[k-1,d])**2
				break
	for k in keys:
		for d in range(0, D):
			Var[k-1,d] = Var[k-1,d] / counter[k]

	return p_y, Mean, Var

def applyNaiveBayes(x, p_y, Mean, Var):
	""" Apply naive base with a given prior
		and conditional probabilites.
	"""
	D = x.size
	K = p_y.size
	# prior
	p_x = p_y.copy()
	# calculate conditional probabilities
	for d in range(0, D):
		for k in range(0, K):
			p_x[k] = p_x[k] * gaussPdf(x[d],Mean[k,d],Var[k,d])

	# find and return class with highest probability
	max_p = 0
	max_k = -1
	for k in range(0, K):
		if p_x[k] > max_p:
			max_p = p_x[k]
			max_k = k

	return max_k + 1

def predictionError(p_y, Mean, Var, X, Y):
	""" Return percentage of wrong classified datapoints. 
		
		X ... Test data, Y ... true labels
	"""
	pred = []
	for x in X:
		pred.append(applyNaiveBayes(x, p_y, Mean, Var))

	pred = np.array(pred)
	
	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y)
	return cnt_wrng / np.size(pred)

# Prepare training data
data = np.loadtxt('wine-train.txt')
X = data[:,1:]
Y = data[:,0]

# train naive bayes with different alpha values
p_y, Mean, Var = trainNaiveBayes(X, Y)
	
# Apply naive bayes on training data
print('Application on training data:')
predEr = predictionError(p_y, Mean, Var, X, Y)
print('Prediction Error: ', predEr)

# Prepare test data
data = np.loadtxt('wine-test.txt')
X_test = data[:,1:]
Y_test = data[:,0]

# Apply naive bayes on test data
print('Application on test data:')
predEr = predictionError(p_y, Mean, Var, X_test, Y_test)
print('Prediction Error: ', predEr)
