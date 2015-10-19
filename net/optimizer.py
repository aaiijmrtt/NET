'''
	Module containing Optimizers and Hyperoptimizers.
	Classes embody Parameter Optimization Algorithms,
	and Hyperparameter optimization Algorithms.
'''
import numpy, copy
from . import error

class Optimizer:
	'''
		Optimizer Class
	'''
	def __init__(self, net, trainingset, testingset, validationset = None, criterion = None):
		'''
			Constructor
			: param net : net whose parameters are to be optimized
			: param trainingset : supervised data set used for training
			: param testingset : supervised data set used for testing
			: param validationset : supervised data set used for validation
			: param criterion : criterion used to quantify error
		'''
		self.net = net
		self.trainingset = trainingset
		self.validationset = validationset if validationset is not None else testingset # default set to testing set
		self.testingset = testingset
		self.criterion = criterion if criterion is not None else error.MeanSquared.compute # default set to half mean squared
		self.error = None

	def train(self, batch = 1, iterations = 1):
		'''
			Method to optimize parameters
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
		'''
		self.net.timingsetup()
		self.net.trainingsetup()
		for i in range(iterations):
			for j in range(len(self.trainingset)):
				if j % batch == 0:
					self.net.updateweights()
					self.net.accumulatingsetup()
					for k in range(j, min(j + batch, len(self.trainingset))):
						self.net.accumulate(self.trainingset[k][0])
					self.net.normalize()
				self.net.feedforward(self.trainingset[j][0])
				self.net.backpropagate(self.trainingset[j][1])

	def validate(self, classification = None):
		'''
			Method to validate training
			: param classification : parameter to control whether task is classification
			: returns : vectorized error
		'''
		classification = classification if classification is not None else False
		self.net.timingsetup()
		self.net.testingsetup()
		if classification:
			self.error = numpy.array([[float(len(self.validationset))]])
			for inputvector, outputvector in self.validationset:
				self.error[0][0] -= outputvector[numpy.argmax(self.net.feedforward(inputvector))][0]
		else:
			self.error = numpy.zeros((self.net.outputs, 1), dtype = float)
			for inputvector, outputvector in self.validationset:
				self.error = numpy.add(self.error, self.criterion(self.net.feedforward(inputvector), outputvector))
		self.error = numpy.divide(self.error, len(self.validationset))
		return self.error

	def test(self, classification = None):
		'''
			Method to test training
			: param classification : parameter to control whether task is classification
			: returns : vectorized error
		'''
		classification = classification if classification is not None else False
		self.net.timingsetup()
		self.net.testingsetup()
		if classification:
			self.error = numpy.array([[float(len(self.testingset))]])
			for inputvector, outputvector in self.testingset:
				self.error[0][0] -= outputvector[numpy.argmax(self.net.feedforward(inputvector))][0]
		else:
			self.error = numpy.zeros((self.net.outputs, 1), dtype = float)
			for inputvector, outputvector in self.testingset:
				self.error = numpy.add(self.error, self.criterion(self.net.feedforward(inputvector), outputvector))
		self.error = numpy.divide(self.error, len(self.testingset))
		return self.error

class Hyperoptimizer(Optimizer):
	'''
		Hyperoptimizer Class
	'''
	def __init__(self, net, trainingset, testingset, validationset = None, criterion = None, hypercriterion = None):
		'''
			Constructor
			: param net : net whose parameters are to be optimized
			: param trainingset : supervised data set used for training
			: param testingset : supervised data set used for testing
			: param validationset : supervised data set used for validation
			: param criterion : criterion used to quantify error
			: param hypercriterion : criterion used to scalarize vectorized error
		'''
		Optimizer.__init__(self, net, trainingset, testingset, validationset, criterion)
		self.hypercriterion = hypercriterion if hypercriterion is not None else numpy.sum # default set to sum

	def gridsearch(self, hyperparameters, batch = 1, iterations = 1):
		'''
			Method to optimize hyperparameters by Grid Search
			: param hyperparameters : initial values of hyperparameters to be optimized. eg. [('applyvelocity', [.3, .5]), ('applylearningrate', [.025, .05])]
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
		'''
		backupnet = copy.deepcopy(self.net)
		indices = [0 for i in range(len(hyperparameters))]
		bestindices = [0 for i in range(len(hyperparameters))]
		limits = [len(hyperparameters[i][1]) for i in range(len(hyperparameters))]
		besterror = float('inf')
		bestnet = backupnet

		while not(indices[len(hyperparameters) - 1] == limits[len(hyperparameters) - 1]):
			for i in range(len(hyperparameters)):
				getattr(self.net, hyperparameters[i][0])(hyperparameters[i][1][indices[i]])
			self.train(batch, iterations)
			error = self.hypercriterion(self.validate())
			if error < besterror:
				besterror = error
				bestindices = copy.deepcopy(indices)
				bestnet = copy.deepcopy(self.net)

			indices[0] += 1
			for i in range(len(indices) - 1):
				if indices[i] == limits[i]:
					indices[i + 1] += 1
					indices[i] = 0
				else:
					break
			self.net = copy.deepcopy(backupnet)

		self.net = copy.deepcopy(bestnet)
		return [hyperparameters[i][1][bestindices[i]] for i in range(len(hyperparameters))]

	def NelderMead(self, hyperparameters, batch = 1, iterations = 1, alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5, threshold = 0.05, hyperiterations = 10): # defaults set
		'''
			Method to optimize hyperparameters by Nelder Mead Algorithm
			: param hyperparameters : initial values of hyperparameters to be optimized. eg. [('applyvelocity', 'applylearningrate'), [.5, .025], [.3, .05], [.1, .025]]
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param alpha : Nelder Mead Algorithm reflection parameter
			: param gamma : Nelder Mead Algorithm expansion parameter
			: param rho : Nelder Mead Algorithm contraction parameter
			: param sigma : Nelder Mead Algorithm reduction parameter
			: param threshold : distance from centroid threshold for termination
			: param hyperiterations : hyperiteration threshold for termination
		'''
		def geterror(self, dimensions, hyperparameters, point, batch, iterations):
			for i in range(dimensions):
				getattr(self.net, hyperparameters[0][i])(point[i])
			self.train(batch, iterations)
			return self.hypercriterion(self.validate())

		backupnet = copy.deepcopy(self.net)
		dimensions = len(hyperparameters[0])
		simplex = [numpy.reshape(hyperparameters[i], (dimensions)) for i in range(1, len(hyperparameters))]
		costs = list()
		besterror = float('inf')
		bestnet = copy.deepcopy(self.net)

		for point in simplex:
			error = geterror(self, dimensions, hyperparameters, point, batch, iterations)
			if error < besterror:
				besterror = error
				bestnet = copy.deepcopy(self.net)
			costs.append(error)
			self.net = copy.deepcopy(backupnet)

		for iteration in range(hyperiterations):
			costs, simplex = zip(*sorted(zip(costs, simplex), key = lambda x: x[0]))
			costs, simplex = list(costs), list(simplex)

			centroid = numpy.divide(numpy.sum(simplex, axis = 0), dimensions)
			if max(numpy.linalg.norm(numpy.subtract(centroid, point)) for point in simplex) < threshold:
				break

			reflectedpoint = numpy.add(centroid, numpy.multiply(alpha, numpy.subtract(centroid, simplex[-1])))
			reflectederror = geterror(self, dimensions, hyperparameters, reflectedpoint, batch, iterations)
			if reflectederror < besterror:
				besterror = reflectederror
				bestnet = copy.deepcopy(self.net)
			self.net = copy.deepcopy(backupnet)

			if costs[0] <= reflectederror < costs[-2]:
				simplex[-1] = reflectedpoint
				costs[-1] = reflectederror

			elif reflectederror < costs[0]:
				expandedpoint = numpy.add(centroid, numpy.multiply(gamma, numpy.subtract(centroid, simplex[-1])))
				expandederror = geterror(self, dimensions, hyperparameters, expandedpoint, batch, iterations)
				if expandederror < besterror:
					besterror = expandederror
					bestnet = copy.deepcopy(self.net)

				if expandederror < reflectederror:
					simplex[-1] = expandedpoint
					costs[-1] = expandederror
				else:
					simplex[-1] = reflectedpoint
					costs[-1] = reflectederror
				self.net = copy.deepcopy(backupnet)

			else:
				if reflectederror < costs[-1]:
					contractedpoint = numpy.add(centroid, numpy.multiply(rho, numpy.subtract(centroid, simplex[-1])))
				else:
					contractedpoint = numpy.add(centroid, numpy.multiply(rho, numpy.subtract(simplex[-1], centroid)))
				contractederror = geterror(self, dimensions, hyperparameters, contractedpoint, batch, iterations)

				if contractederror < besterror:
					besterror = contractederror
					bestnet = copy.deepcopy(self.net)
				self.net = copy.deepcopy(backupnet)

				if contractederror < costs[-1]:
					simplex[-1] = contractedpoint
					costs[-1] = contractederror
				else:
					for i in range(1, len(simplex)):
						simplex[i] = numpy.add(simplex[0], numpy.multiply(sigma, numpy.subtract(simplex[i], simplex[0])))
						costs[i] = geterror(self, dimensions, hyperparameters, simplex[i], batch, iterations)
						if costs[i] < besterror:
							besterror = costs[i]
							bestnet = copy.deepcopy(self.net)
						self.net = copy.deepcopy(backupnet)

		self.net = copy.deepcopy(bestnet)
		return [(cost, point.tolist()) for cost, point in zip(costs, simplex)]
