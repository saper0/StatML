#######################################################################
# Author: Lukas Gosch
# Date: 04.11.2018
# Description: This code implements two different methods to 
#			   solve the linear SVM optimization problem for the wine
#			   dataset:
#			   I) Stochastic Subgradient Method for the Primal Problem
#			   II) Stochastic Dual Coordinate Descent for the Dual 
#				   Problem
#			   Extention to multi-class prediction is done via the 
#			   one-versus-all approach.
#
#			   Details:
#			   I) Is equal to the code developed for exercises 3.
#			      The implementation follows roughly the pseudocode
#			      in chapter 15.5 S213 in "Understanding Machine
#				  Learning: From Theory to Algorithms" (Shwartz)
#				  titled SGD for Solving Soft-SVM. 
#			   II)The implementation follows the pseudocode of slide 
#				  33 of lecture 4, therefore the bias term is treated 
#				  implicit. 	    
#######################################################################

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def trainMultiSVM(X, Y, T, lmbda=1, dual = False):
	""" Train a multiclass SVM using either I) stochastic subgradient 
		method or II) stochastic coordinate dual ascent and a 
		one-versus-all approach. 

		X ... training feature vectors
		Y ... true labels element of [1, K]
		T ... iterations
		lmbda ... coefficient controlling how strong a wrong
				   classification should be weighted
				   (small lmbda => high weight)
		dual ... false: solve primal problem with I) 
				 true: solve dual problem with II)
	"""
	labels = [int(k) for k in set(Y)]
	# one - versus - all
	h_l = []
	eval_l = []
	dual_l = []
	for label in labels:
		# construct binary labeling
		Y_bnry = [1 if y == label else -1 for y in Y]
		# train binary SVM by either solving dual or primal problem
		if dual:
			w, b, w_l, b_l, a, a_l = trainSVMSCDA(X, Y_bnry, T, lmbda)
		else:
			w, b, w_l, b_l = trainSVMSGD(X, Y_bnry, T, lmbda)
		# add SVM to binary predictor list
		h_l.append((w,b))
		# store for later evaluation of learning
		eval_l.append((w_l,b_l))
		if dual:
			dual_l.append((a,a_l))
	if dual:
		return h_l, eval_l, dual_l
	return h_l, eval_l

def trainSVMSCDA(X, Y, T, lmbda=1):
	""" Train a linear SVM using the stochastic coordinate dual ascent. 
		
		Uses implicit bias.

		X ... training feature vectors
		Y ... true labels element of {-1, 1}
		T ... iterations
		lmbda ... coefficient controlling how strong a wrong
				   classification should be weighted
				   (small lmbda => high weight)
	"""
	N = X.shape[0]
	C = 1/(N*lmbda) # regularize with lambda instead of C
	d = X.shape[1]
	alpha = np.zeros((N,))
	w = np.zeros((d+1,))
	w_l = []
	b_l = []
	a_l = []
	for t in range(1, T+1):
		# choose (uniformly distributed) random data point
		i = np.random.randint(0, N)
		x_i = np.append(X[i,:],1)
		y_i = Y[i]
		# calculate alpha update
		delta = (1 - y_i * np.inner(w, x_i)) / np.inner(x_i, x_i)
		# perform update
		alpha_new = alpha[i] + delta
		if alpha_new < 0:
			alpha_new = 0
		if alpha_new > C:
			alpha_new = C
		# update weight vector
		w = w + (alpha_new - alpha[i])*y_i*x_i
		alpha[i] = alpha_new
		# store weight vector for later evaluation
		w_l.append(w[:d])
		b_l.append(w[d])
		a_l.append(np.copy(alpha))

	return w[:d], w[d], w_l, b_l, alpha, a_l


def trainSVMSGD(X, Y, T, lmbda=1):
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

def calcPrimal(lmbda, w, b, X, Y_bnry, implicit = False):
	""" Calculate (primal) objective value of a given binary SVM. 
		
		implicit ... was the bias term treated implicitly or not?
	"""
	avg_err = 0
	for i in range(0, X.shape[0]):
		err = 1 - Y_bnry[i] * (np.inner(w, X[i,]) + b)
		if err > 0:
			avg_err = avg_err + err
	avg_err = 1/X.shape[0] * avg_err

	if implicit:
		return lmbda / 2 * (np.inner(w, w) + b*b) + avg_err 
	return lmbda / 2 * np.inner(w, w) + avg_err

def calcDual(lmbda, alpha, X, Y_bnry):
	""" Calculate value of dual problem objective. """
	N = X.shape[0]
	term1 = 0
	term2 = 0
	for i in range(0, N):
		# sum over all lagrangian multiplier
		term1 = term1 + alpha[i]
		for j in range(0, N):
			term2 = term2 + alpha[i]*alpha[j]*Y_bnry[i]*Y_bnry[j]*np.inner(X[i,], X[j,])
	return term1 - 1/(2*lmbda) * term2

def evalLearning(h_l, eval_l, lmbda, T, X, Y, dual_l, implicit = False, dual = False):
	""" Return list of SVM's objective values and the euclidian
		distances to the optimum for each iteration and binary SVM.

		implicit ... was the bias term treated implicitly or not?
		dual ... if true, calculates objective of dual problem not of primal
				 one, if dual is true, requiers dual_l to be not empty
	"""
	labels = [int(k) for k in set(Y)]
	obj_vals = []
	dists_opt = []
	for label in labels:
		# construct binary labeling
		Y_bnry = [1 if y == label else -1 for y in Y]
		# evaluate primal problem
		if not dual:
			# retrieve optimal classifier
			w, b = h_l[label-1]
			# retrieve w, b during the learning process
			w_l, b_l = eval_l[label-1]
			# calculate SVM's objective value
			obj_val = []
			for i in range(0, T):
				obj_val.append(calcPrimal(lmbda, w_l[i], b_l[i], X, Y_bnry, implicit))
			obj_vals.append(obj_val)
			# calculate distance to optimal value
			dist_opt = []
			for i in range(0, T):
				dist_opt.append(np.linalg.norm(w - w_l[i]))
		# evaluate dual problem
		if dual:
			# retrieve langrangien multipliers
			alpha, a_l = dual_l[label-1] 
			# calculate SVM's (dual) objective value
			obj_val = []
			for i in range(0, T):
				obj_val.append(calcDual(lmbda, a_l[i], X, Y_bnry))
			obj_vals.append(obj_val)
			# calculate distance to optimal value
			dist_opt = []
			for i in range(0, T):
				dist_opt.append(np.linalg.norm(alpha - a_l[i]))

		dists_opt.append(dist_opt)

	return obj_vals, dists_opt

def main(argv):
	parser = argparse.ArgumentParser(description='Call with --dual if the dual optimization problem should be solved instead of the primal.')
	# Choose if dual problem should be solved instead of primal
	parser.add_argument('--dual', action='store_true', default=False)
	# Choose if dual objective should be evaluated instead of primal
	parser.add_argument('--eval-dual', action='store_true', default=False)
	args = parser.parse_args(argv[1:])

	# Prepare training data
	data = np.loadtxt('wine-train.txt')
	X = data[:,1:]
	# scale to zero mean and unit variance
	X = preprocessing.scale(X)
	Y = data[:,0]

	# Train SVM
	np.random.seed(1)
	if args.dual:
		#lmbda for dual
		lmbda = 1
	else:
		#lmbda for SGD
		lmbda=0.27
	#for lmbda in np.linspace(0,1,50):
	T = 40
	if args.dual:
		h_l, eval_l, dual_l = trainMultiSVM(X, Y, T, lmbda, args.dual)
	else:
		h_l, eval_l = trainMultiSVM(X, Y, T, lmbda, args.dual)

	print(h_l)
	# Apply SVM on training data
	print('Lambda: ', lmbda)
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

	# Evaluate data for report, if dual problem was solved, bias is implicit
	if not args.eval_dual:
		obj_vals, dists_opt = evalLearning(h_l, eval_l, lmbda, T, X, Y, None, implicit = args.dual)
	else:
		obj_vals, dists_opt = evalLearning(h_l, eval_l, lmbda, T, X, Y, dual_l, implicit = args.dual, dual=True)

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
	second.plot(range(1, T+1),dists_opt[0], 'o-',color='g', label='Binary SVM Label 1 vs all')
	second.plot(range(1, T+1),dists_opt[1], 'o-',color='y', label='Binary SVM Label 2 vs all')
	second.plot(range(1, T+1),dists_opt[2], 'o-',color='b', label='Binary SVM Label 3 vs all')
	second.set_ylabel('Euclidean distance to the optimum')
	second.set_xlabel('Iterations')
	second.legend()

	plt.show()

if __name__ == "__main__":
	main(sys.argv)






