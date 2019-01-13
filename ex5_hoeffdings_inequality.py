#######################################################################
# Author: Lukas Gosch
# Date: 11.11.2018
# Usage: See python ex5.py --help
# Description: Compute Hoeffding's inequality, estimate left hand side
#			   of the bound based on differently parametrized random
#			   experiments.
#######################################################################
import sys
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
	parser = argparse.ArgumentParser(description='Call with -ab for a) & b) and with -cd for c) & d)')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('-ab', action='store_true', default=False)
	group.add_argument('-cd', action='store_true', default=False)
	args = parser.parse_args(argv[1:])

	np.random.seed(1)

	# define number of random experiments
	n = 100

	# define bernoulli distribution
	p_l = [0.5, 0.1]

	# define error and sample size
	if args.ab:
		epsilon = 0.25
		m_l = [i for i in range(5,51,5)]
		m_l.insert(0,1)
	
	if args.cd:
		epsilon = 0.05
		m_l = [i for i in range(50,501,50)]
		m_l.insert(0,10)

	# estimate left hand side
	exp = []
	for p in p_l:
		lhs_l = []
		for m in m_l:
			lhs = 0
			for i in range(0, n):
				# draw random samples i.i.d. and calc. empirical avg.
				e_avg = sum(np.random.binomial(1,p,m)) / m
				if math.fabs(e_avg - p) > epsilon:
					lhs = lhs + 1
			# empirically estimate left hand side
			lhs = lhs / n
			lhs_l.append(lhs)
		exp.append(lhs_l)

	# calculate bound
	rhs_l = 2*np.exp(-2*epsilon*epsilon*np.array(m_l))

	# plot values in a graph with sample size on the x-axis and bound
	# as well as estimate on the y axis
	plt.title('Experiment Results')
	plt.plot(m_l, exp[0], 'o-', color='g', label='LHS: empirical estimate for mu = 0.5')
	plt.plot(m_l, exp[1], 'o-', color='b', label='LHS: empirical estimate for mu = 0.1')
	plt.plot(m_l, rhs_l, 'x-', color='k', label='RHS: bound value (Hoeffding\'s inequality)')
	plt.plot()
	plt.ylabel('Probability')
	plt.xlabel('Sample size m')
	plt.xticks(m_l)
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main(sys.argv)
