#######################################################################
# Author: Lukas Gosch
# Date: 20.10.2018
# Description: Implementation of a Bernoulli Naive Bayes for the 
#			   generated binary dataset.
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

def laplaceSmoothing(N, n, alpha=1, Z=2):
	""" Calculate laplacian smoothing. 
		N ... total count
		n ... conditional count
	"""
	return (alpha + n)/(N + Z*alpha)

def trainNaiveBayes(X, Y, alpha=1):
	""" Train a naive bayes classifier. 

		Trains a naive bayes classifier in the case of binary feature 
		vectors and two classes. Uses a bernoulli model and laplacian 
		smoothing.

		X ... training feature vectors
		Y ... true labels
		alpha ... parameter for laplacian smoothing (Z=2)
				  (default: Laplace's rule of succession alpha=1)
	"""
	# Calculate Prior p(y)
	N = Y.size
	N_1 = 0
	for i in range(0, N):
		if Y[i] == 1:
			N_1 = N_1 + 1
	N_0 = N - N_1 # N_0 is count of y == -1
	p_y = N_1 / N

	# Calculate probability matrix storing probability
	# of class y_k generating x_i
	d = X.shape[1] 
	Theta = np.zeros((2, d))
	for i in range(0, d):
		count_x_1 = 0
		count_x_0 = 0
		for n in range(0, N):
			if Y[n] == 1:
				if X[n, i] == 1:
					count_x_1 = count_x_1 + 1
			if Y[n] == -1:
				if X[n, i] == 1:
					count_x_0 = count_x_0 + 1
		Theta[1, i] = laplaceSmoothing(N_1, count_x_1, alpha)
		Theta[0, i] = laplaceSmoothing(N_0, count_x_0, alpha)
	return p_y, Theta

def applyNaiveBayes(x, p_y, Theta):
	""" Apply naive base with a given prior
		and conditional probabilites.

		If tie, returns 1
	"""
	d = x.size
	# prior
	p_1 = p_y
	p_0 = 1 - p_y
	# calculate conditional probabilities
	for i in range(0,d):
		if x[i] == 1:
			p_1 = p_1 * Theta[1, i]
			p_0 = p_0 * Theta[0, i]
		else:
			p_1 = p_1 * (1 - Theta[1,i])
			p_0 = p_0 * (1 - Theta[0,i])
	# return class with highest probability
	if p_0 > p_1:
		return -1
	return 1

def predictionError(p_y, Theta, X, Y):
	""" Return percentage of wronge classified datapoints. 
		
		X ... Test data, Y ... true labels
	"""
	pred = []
	for x in X:
		pred.append(applyNaiveBayes(x, p_y, Theta))

	pred = np.array(pred)
	
	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y)
	return cnt_wrng / np.size(pred)

def main(argv):
	parser = argparse.ArgumentParser(description='Call with --swapped if 9th data point should have label -1 instead of +1.')
	# Choose if 9th training point should swap label
	parser.add_argument('--swapped', action='store_true', default=False)
	args = parser.parse_args(argv[1:])
	
	# Prepare training data
	X, Y = genTrainingData(args.swapped)
	X = np.array(X)
	Y = np.array(Y)

	# train naive bayes with different alpha values
	models = []
	alphas = np.arange(0.1,1.01,0.1)
	for alpha in alphas:
		p_y, Theta = trainNaiveBayes(X, Y, alpha)
		models.append((alpha, p_y, Theta))

	# Apply naive bayes on training data
	print('Application on training data:')
	alpha_vs_train = []
	for model in models:
		alpha = model[0]
		p_y = model[1]
		Theta = model[2]
		predEr = predictionError(p_y, Theta, X, Y)
		alpha_vs_train.append(predEr)
		print('Prediction Error: ', predEr, ' Alpha: ', alpha)

	# Prepare test data
	X, Y = genTestData()
	X = np.array(X)
	Y = np.array(Y)

	# Apply naive bayes on test data
	print('Application on test data:')
	alpha_vs_test = []
	for model in models:
		alpha = model[0]
		p_y = model[1]
		Theta = model[2]
		predEr = predictionError(p_y, Theta, X, Y)
		alpha_vs_test.append(predEr)
		print('Prediction Error: ', predEr, ' Alpha: ', alpha)

	# Plot alpha vs training/test error
	plt.plot(alphas, alpha_vs_train, 'o-',color='g', label='Training error')
	plt.plot(alphas, alpha_vs_test, 'o-',color='y', label='Test error')
	plt.title('Alpha vs Error')
	plt.ylabel('Prediction Error')
	plt.xlabel('Alpha value')
	#plt.xticks(alphas)

	plt.show()

if __name__ == "__main__":
	main(sys.argv)