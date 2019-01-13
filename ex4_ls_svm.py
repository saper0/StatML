#######################################################################
# Author: Lukas Gosch
# Date: 2.11.2018
# Description: Implementation of Least Squares SVM.
#######################################################################

import numpy as np
import matplotlib.pyplot as plt

def split(X, Y):
	""" Split (X, Y) in approximatly equal sized train/validation set. """
	N = X.shape[0]
	N_trn = int(N/2) + 1
	# draw samples without replacement from N
	trn_i = np.random.choice(N, N_trn, replace=False)
	trn_i = np.sort(trn_i)
	val_i = np.delete(np.arange(N), trn_i)
	# create split
	X_trn = X[trn_i,:]
	Y_trn = Y[trn_i]
	X_val = X[val_i,:]
	Y_val = Y[val_i]

	return X_trn, Y_trn, X_val, Y_val

def trainLSSVM(X, Y, lmbda=0):
	""" Train a Least Squares SVM with given regularization lmbda. """
	D = X.shape[1]
	X_T = X.T
	# directly calculate w
	inv = np.linalg.inv(np.matmul(X_T, X) + lmbda * np.identity(D))
	return np.matmul(inv, X_T.dot(Y))

def applyLSSVM(x, w):
	""" Apply a trained LS-SVM for binary classification of datapoint x. """
	if np.inner(w, x) > 0:
		return 1
	return -1

def predictionError(X, Y, w):
	""" Return percentage of wrong classified datapoints. 
		
		X ... Test data, Y ... true labels
		w ... weight vector SVM, b .. bias term SVM
	"""
	pred = []
	for x in X:
		pred.append(applyLSSVM(x, w))

	pred = np.array(pred)

	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y)
	return cnt_wrng / np.size(pred)

X = np.loadtxt('Xtrain.txt')
Y = np.loadtxt('Ytrain.txt')

np.random.seed(1)
X_trn, Y_trn, X_val, Y_val = split(X, Y)

# train model without regularization
w = trainLSSVM(X_trn, Y_trn, lmbda = 0)
print("Results no regularization:")
p_e = predictionError(X_trn, Y_trn, w)
print("Prediction Error: ", p_e, " 0/1-loss: ", p_e*X_trn.shape[0])
p_e = predictionError(X_val, Y_val, w)
print("Prediction Error: ", p_e, " 0/1-loss: ", p_e*X_val.shape[0])

# train model for different lambda values evenly spaced on a log scale
lmbda_l = np.logspace(-20, 20, num = 41, base = 10)
trn_e = []
val_e = []
for i in range(0, lmbda_l.size):
	w = trainLSSVM(X_trn, Y_trn, lmbda = lmbda_l[i])
	trn_e.append(predictionError(X_trn, Y_trn, w))
	val_e.append(predictionError(X_val, Y_val, w))

# plot regularization results (if wanted)
"""
plt.title('Training and validation error for different regularizations')
plt.plot(lmbda_l, trn_e, 'o-', color='g', label='Training Set')
plt.plot(lmbda_l, val_e, 'o-', color='b', label='Validation Set')
plt.ylabel('Prediction Error')
plt.xlabel('Value of regularization parameter')
plt.xscale(value="log")
plt.xticks(np.logspace(-20, 20, num = 11, base = 10))
plt.legend()
plt.show()
"""

# extract hyperparameter for which model had lowest validation error
min_e = min(val_e)
min_i = val_e.index(min_e)
lmbda = lmbda_l[min_i]
# if lmbda = 0 performed best
if min_e > p_e:
	min_e = p_e
	lmbda = 0

print("Best lambda: ", lmbda, " Validation Error: ", min_e)
w = trainLSSVM(X, Y, lmbda = lmbda)

# load test data
X = np.loadtxt('Xtest.txt')
Y = np.loadtxt('Ytest.txt')

print("Prediction Error on Test Data: ", predictionError(X, Y, w))








