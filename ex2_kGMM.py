#######################################################################
# Author: Lukas Gosch
# Date: 21.10.2018
# Description: Implementation of a k-component Gaussian Mixture Model 
#######################################################################

import math
import numpy as np


def genXORData(N = 200):
	""" Generate XOR Dataset with N entries. 

		N should be divisible by 4.
	"""
	n = int(N / 4)
	# left up
	x_lu = np.random.uniform(-1, 0, n)
	y_lu = np.random.uniform(0, 1, n)
	# right down
	x_rd = np.random.uniform(0, 1, n)
	y_rd = np.random.uniform(-1, 0, n)
	# right up
	x_ru = np.random.uniform(0, 1, n)
	y_ru = np.random.uniform(0, 1, n)
	# left down
	x_ld = np.random.uniform(-1, 0, n)
	y_ld = np.random.uniform(-1, 0, n)

	x_cord = np.append(x_lu, [x_rd, x_ru, x_ld])
	y_cord = np.append(y_lu, [y_rd, y_ru, y_ld])

	# labels
	y = np.ones((N,))
	y[:2*n] = -1

	# create train/test split
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	train_ind = np.random.choice(N, int(N/2), replace=False)
	train_ind = np.sort(train_ind)
	test_ind = np.delete(np.arange(N), train_ind)
	for i in train_ind:
		X_train.append([x_cord[i], y_cord[i]])
		Y_train.append(y[i])
	for i in test_ind:
		X_test.append([x_cord[i], y_cord[i]])
		Y_test.append(y[i])

	return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

def trainGMM(X, K, mean, cov, mix, minimal_change=1e-6):
	""" Train a Gaussian Mixture Model with K components.
		
		X ... training data (of a certain class)
		mean ... initialized means of the k components
		cov ... initialized covariance matrices of the k components
		mix ... initialized mixing coefficients of the k components
		minimal_change ... minimal change in the log-likelihood function
						   for each iteration, otherwise converged
	"""
	N = X.shape[0]
	gammas = np.zeros((N,K))
	converged = False # not used just for readability
	log_likelihood_old = 0
	while not converged:
		#### Expectation Step ####
		# Calculate Responsibilities
		for k in range(0, K):
			cov_inv = np.linalg.inv(cov[k])
			cov_det = np.linalg.det(cov[k])
			for n in range(0, N):
				# evaluate coefficient for exponent
				v = X[n] - mean[k]
				coeff = v.dot(cov_inv)
				coeff = coeff.dot(v)
				coeff = - 1/2 * coeff
				# calculate mixing coefficient * p(x|mu,var)
				gammas[n,k] = mix[k] / math.sqrt(2 * math.pi * cov_det) * math.exp(coeff)
		# Evaluate log-likelihood (of previous parameters)
		Z = np.sum(gammas, axis=1)
		log_likelihood = np.sum(np.log(Z))
		# If the change in the log likelihood from the previous
		# update is smaller then a certain threshold, algorithm
		# has converged.
		if abs(log_likelihood - log_likelihood_old) < minimal_change:
			return mix, mean, cov
		# Normalize
		gammas = np.multiply(gammas, (1 / Z)[:, np.newaxis])

		#### Maximization Step ####
		# Update Mean
		D = X.shape[1]
		N_gamma = np.sum(gammas, axis=0)
		mean = np.zeros((K,D))
		for k in range(0, K):
			for n in range(0, N):
				mean[k] = mean[k] + gammas[n,k] * X[n]
			mean[k] = 1 / N_gamma[k] * mean[k]
		# Update Covariance Matrix
		cov = np.zeros((K,D,D))
		for k in range(0, K):
			for n in range(0, N):
				v = X[n] - mean[k]
				cov[k] = cov[k] + gammas[n,k] * np.outer(v, v)
			cov[k] = 1 / N_gamma[k] * cov[k]
		# Update mixture coefficients
		for k in range(0, K):
			mix[k] = N_gamma[k] / N

		# For evaluation next interation
		log_likelihood_old = log_likelihood

def applyGMMs(x, GMM_list):
	""" Return result of general bayes classifier based on the
		given gaussian mixture models (incl. estimates for p(y)).
		Assumes binary classes -1 and 1.
	"""
	p_x = np.zeros((2,))
	for i in range(0, len(GMM_list)):
		model = GMM_list[i]
		p_y = model[0]
		mix = model[1]
		mean = model[2]
		cov = model[3]
		# calculate p(x|y) for the chosen Gaussian Mixture Model
		K = len(mix)
		for k in range(0, K):
			cov_inv = np.linalg.inv(cov[k])
			cov_det = np.linalg.det(cov[k])
			# evaluate coefficient for exponent
			v = x - mean[k]
			coeff = v.dot(cov_inv)
			coeff = coeff.dot(v)
			coeff = - 1/2 * coeff

			# update p(x|y) with component k
			p_x[i] = p_x[i] + mix[k] / math.sqrt(2 * math.pi * cov_det) * math.exp(coeff)

			# calculate p(x,y) = p(x|y)*p(y)
			p_x[i] = p_x[i] * p_y

	# return most probable class
	if p_x[0] > p_x[1]:
		return 1
	return -1

def predictionError(X, Y, GMM_list):
	""" Return percentage of wrong classified datapoints. 
		
		X ... Test data, Y ... true labels
	"""
	pred = []
	for x in X:
		pred.append(applyGMMs(x, GMM_list))

	pred = np.array(pred)
	
	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y)
	return cnt_wrng / np.size(pred)

np.random.seed(1)
minimal_change=1e-14

# generate dataset and training/test split
X_train, Y_train, X_test, Y_test = genXORData()

# separate training data
X_lu = []
X_rd = []
X_ru = []
X_ld = []
N = Y_train.size
for i in range(0, N):
	if Y_train[i] == 1:
		if X_train[i,0] > 0:
			X_ru.append(X_train[i])
		else:
			X_ld.append(X_train[i])
	else:
		if X_train[i,0] > 0:
			X_rd.append(X_train[i])
		else:
			X_lu.append(X_train[i])

# initializations for y = +1
X_ru = np.array(X_ru)
X_ld = np.array(X_ld)
mean = [X_ru.mean(axis=0), X_ld.mean(axis=0)]
cov = [np.cov(X_ru, rowvar=False, bias=True), np.cov(X_ld, rowvar=False, bias=True)]
mix = [1/2, 1/2]

# train GMM with two components for y = +1
X_pos = np.append(X_ru, X_ld, axis=0)
mix1, mean1, cov1 = trainGMM(X_pos, 2, mean, cov, mix, minimal_change)

# initializations for y = -1
X_lu = np.array(X_lu)
X_rd = np.array(X_rd)
mean = [X_lu.mean(axis=0), X_rd.mean(axis=0)]
cov = [np.cov(X_lu, rowvar=False, bias=True), np.cov(X_rd, rowvar=False, bias=True)]
mix = [1/2, 1/2]

# train GMM with two components for y = -1
X_neg = np.append(X_lu, X_rd, axis=0)
mix_1, mean_1, cov_1 = trainGMM(X_neg, 2, mean, cov, mix, minimal_change)

# estimate p(y)
p_y = np.zeros((2,))
p_y[0] = np.sum(Y_train == 1)
p_y[1] = N - p_y[0]

model1 = (p_y[0], mix1, mean1, cov1)
model_1 = (p_y[1], mix_1, mean_1, cov_1)

print("--Prediction Errors--")
print("Training Set: ", predictionError(X_train, Y_train, [model1, model_1]))
print("Test Set: ", predictionError(X_test, Y_test, [model1, model_1]))
