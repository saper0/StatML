#######################################################################
# Author: Lukas Gosch
# Date: 20.10.2018
# Description: Implementation of a gaussian discriminant analysis for 
#			   the wine dataset.
#######################################################################

import sys
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import math

def trainGDA(X, Y):
	""" Train a gaussian discriminant analysis classifier. 

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

	# Calculate mean over whole data
	d = X[0].size
	mu = X.mean(axis=0)

	# Calculate shared covariance matrix
	cov = np.cov(X, rowvar=False, bias=True)

	# Calculate class means
	mu_y = []
	for k in range(0,K):
		mu_y.append(np.zeros((d,)))

	for i in range(0,N):
		k = int(Y[i]-1)
		mu_y[k] = mu_y[k] + X[i]

	for k in range(0,K):
		mu_y[k] = 1 / counter[k+1] * mu_y[k]

	# Calculate inverse and determinante of covariance
	# matrix, so to only compute it once
	cov_inv = np.linalg.inv(cov)
	cov_det = np.linalg.det(cov)

	return p_y, mu_y, mu, cov_inv, cov_det

def applyGDA(x, p_y, mu_y, mu, cov_inv, cov_det):
	""" Apply a trained GDA with K classes to a datapoint x. 
		
		p_y ... priors for the K classes
		mu_y ... list storing the K class means
		mu ... total mean over all classes
		cov_inv ... inverse (total) covariance matrix
		cov_det ... determinant of (total) covariance matrix
	"""
	K = p_y.size
	p_x = np.zeros((K,))
	for k in range(0, K):
		# evaluate coefficient for exponent
		v = x - mu_y[k]
		coeff = v.dot(cov_inv)
		coeff = coeff.dot(v)
		coeff = - 1/2 * coeff

		# calculate p(x|y)
		p_x[k] = 1 / math.sqrt(2 * math.pi * cov_det) * math.exp(coeff)

		# calculate p(x,y) = p(x|y)*p(y)
		p_x[k] = p_x[k] * p_y[k]

	# return y for which p(x,y) is max
	max_k = -1
	max_p = 0
	for k in range(0,K):
		if p_x[k] > max_p:
			max_p = p_x[k]
			max_k = k

	return max_k + 1


def predictionError(p_y, mu_y, mu, cov_inv, cov_det, X, Y):
	""" Return percentage of wrong classified datapoints. 
		
		X ... Test data, Y ... true labels
	"""
	pred = []
	for x in X:
		pred.append(applyGDA(x, p_y, mu_y, mu, cov_inv, cov_det))

	pred = np.array(pred)
	
	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y)
	return cnt_wrng / np.size(pred)

# Prepare training data
data = np.loadtxt('wine-train.txt')
X = data[:,1:]
Y = data[:,0]

# train gda
p_y, mu_y, mu, cov_inv, cov_det = trainGDA(X,Y)
	
# Apply gda on training data
print('Application on training data:')
predEr = predictionError(p_y, mu_y, mu, cov_inv, cov_det, X, Y)
print('Prediction Error: ', predEr)

# Prepare test data
data = np.loadtxt('wine-test.txt')
X_test = data[:,1:]
Y_test = data[:,0]

# Apply gda on test data
print('Application on test data:')
predEr = predictionError(p_y, mu_y, mu, cov_inv, cov_det, X_test, Y_test)
print('Prediction Error: ', predEr)