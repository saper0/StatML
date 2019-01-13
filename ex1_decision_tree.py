#######################################################################
# Author: Lukas Gosch
# Date: 14.10.2018
# Description: Implementation of a decision tree.
#######################################################################

import sys
import argparse
import collections
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

class Node:
	def __init__(self, isLeaf = False):
		self.isLeaf = isLeaf

	def addFeature(self, feature):
		if not self.isNodeLeaf():
			self.feature = feature
		else:
			print('Node is leaf! Cannot add feature to leaf!')

	def checkFeature(self, x):
		""" Check if x should be further processed down on the left or
			right path. False means left, True means right.
		"""
		if not self.isNodeLeaf():
			if x[self.feature] == 0:
				return False
			else:
				return True
		else:
			print('Node is leaf! Connat split based on feature!')

	def addLeft(self, left):
		""" Left direction if feature is 0. """
		if not self.isNodeLeaf():
			self.left = left
		else:
			print('Node is leaf! Cannot add further nodes.')

	def getLeft(self):
		if not self.isNodeLeaf():
			return self.left
		else:
			print('Node is leaf! Has no left node!')

	def addRight(self, right):
		""" Right direction if feature is 1. """
		if not self.isNodeLeaf():
			self.right = right
		else:
			print('Node is leaf! Cannot add further nodes.')

	def getRight(self):
		if not self.isNodeLeaf():
			return self.right
		else:
			print('Node is leaf! Has no right node!')

	def isNodeLeaf(self):
		return self.isLeaf

	def setLeafValue(self, val):
		if self.isNodeLeaf():
			self.value = val
		else:
			print('Node is no leaf! Cannot store a prediction value.')

	def getLeafValue(self):
		if self.isNodeLeaf():
			return self.value
		else:
			print('Node is no leaf! Cannot store a prediction value.')

def decisionTreeTrain(X, Y, features, tie_decision = 1, depth = 1, node_count=0, max_nodes = -1):
	""" Train Decision Tree with given data. 
		
		features represent indices of vectors in X.	
		tie_decision is used if tie or empty leaf
		depth is level/hight of tree
		node_count counts interior nodes already in the tree
		max_nodes -1 means full tree, otherwise upper bound on interior 
		    node count
	"""

	# count occurance of labels
	counter = collections.Counter(Y).most_common()
	if len(counter) == 0:
		# empty node, choose tie decision
		guess = tie_decision
	else:
		# choose most frequent answer in Y
		if len(counter) == 1:
			guess = counter[0][0]
		else:
			# check if tie
			if counter[0][1] == counter[1][1]:
				guess = tie_decision
			else:
				guess = counter[0][0]
	
	if len(counter) <= 1:
		# perfect labeling or empty leaf
		print('Depth: ', depth, ' Perfect Leaf', ' Decision: ', guess)
		leaf = Node(isLeaf = True)
		leaf.setLeafValue(guess)
		return leaf, node_count

	if len(features) == 0:
		# no further features to split
		print('Depth: ', depth, ' Leaf, not features left', ' Decision: ', guess)
		leaf = Node(isLeaf = True)
		leaf.setLeafValue(guess)
		return leaf, node_count

	if max_nodes == node_count:
		# maximum level (complexity) of tree reached
		print('Depth: ', depth, ' Leaf, maximum node count reached.', ' Decision: ', guess)
		leaf = Node(isLeaf = True)
		leaf.setLeafValue(guess)
		return leaf, node_count

	# find best split
	max_no = []
	max_yes = []
	max_score = -1
	max_feature = -1
	for f in features:
		no = [] # store indices of subset of data on which feature = -1
		yes = [] # store indices of subset of data on which feature = 1
		for i in range(0, len(X)):
			if X[i][f] == 0:
				no.append(i)
			else:
				yes.append(i)

		score_f = score(Y[no]) + score(Y[yes])
		if score_f > max_score:
			max_score = score_f
			max_feature = f
			max_no = no
			max_yes = yes

	# exclude used feature
	features = [f for f in features if f != max_feature]
	# create left and right decision tree
	node_count = node_count + 1
	print('Depth: ', depth, 'Node: ', node_count, ' Split feature: ', max_feature)
	left, node_count = decisionTreeTrain(X[max_no], Y[max_no], features, tie_decision, depth + 1, node_count, max_nodes)
	right, node_count = decisionTreeTrain(X[max_yes], Y[max_yes], features, tie_decision, depth + 1, node_count, max_nodes)
	# create and return node
	node = Node(isLeaf = False)
	node.addLeft(left)
	node.addRight(right)
	node.addFeature(max_feature)
	return node, node_count

def score(labels):
	""" Return count of majority label in given list. """

	count_no = 0
	count_yes = 0
	for y in labels:
		if y == -1:
			count_no = count_no + 1
		else:
			count_yes = count_yes + 1

	if count_no > count_yes:
		return count_no
	if count_yes > count_no:
		return count_yes

	# tie
	return count_no

def decisionTreeTest(tree, x):
	if tree.isNodeLeaf():
		return tree.getLeafValue()
	else:
		if not tree.checkFeature(x):
			return decisionTreeTest(tree.getLeft(), x)
		else:
			return decisionTreeTest(tree.getRight(), x)

def predictionError(tree, X, Y):
	""" Return percentage of wronge classified datapoints. """
	pred = []
	""" X ... Test data, Y ... true labels """
	for x in X:
		pred.append(decisionTreeTest(tree, x))

	pred = np.array(pred)
	
	cnt_wrng = np.size(pred) - np.count_nonzero(pred == Y)
	return cnt_wrng / np.size(pred)

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

	# Train decision tree
	print('Full tree: ')
	full_tree, node_count = decisionTreeTrain(X, Y, features)
	trees = []
	for max_nodes in range(0, node_count):
		print('Tree, max_nodes: ', max_nodes)
		trees.append(decisionTreeTrain(X, Y, features, max_nodes=max_nodes))
	trees.append((full_tree, node_count))
	
	# Apply decision tree on training data
	print('Application on training data:')
	complexity_vs_train = []
	for tree in trees:
		predEr = predictionError(tree[0], X, Y)
		complexity_vs_train.append(predEr)
		print('Prediction Error: ', predEr, ' Interior Nodes: ', tree[1])

	# Prepare test data
	X, Y = genTestData()
	X = np.array(X)
	Y = np.array(Y)

	# Apply decision tree on test data
	print('Application on test data:')
	complexity_vs_test = []
	for tree in trees:
		predEr = predictionError(tree[0], X, Y)
		complexity_vs_test.append(predEr)
		print('Prediction Error: ', predEr, ' Interior Nodes: ', tree[1])

	# Plot complexity vs training/test error
	plt.plot(complexity_vs_train, 'o-',color='g', label='Training error')
	plt.plot(complexity_vs_test, 'o-',color='y', label='Test error')
	plt.title('Complexity vs Error')
	plt.ylabel('Prediction Error')
	plt.xlabel('Number of interior nodes')
	plt.xticks(range(0, len(complexity_vs_train)))

	plt.show()

if __name__ == "__main__":
	main(sys.argv)