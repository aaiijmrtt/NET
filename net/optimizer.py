'''
	Module containing Optimizers and Hyperoptimizers.
	Classes embody Parameter Optimization Algorithms,
	and Hyperparameter Optimization Algorithms.
'''
import numpy, copy, dispy
from . import base, configure, error
import data

class Optimizer(base.Net):
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
		self.testingset = testingset
		self.validationset = validationset if validationset is not None else self.testingset # default set to testing set
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
				self.error[0][0] -= outputvector[configure.functions['argmax'](self.net.feedforward(inputvector))][0]
		else:
			self.error = numpy.zeros((self.net.dimensions['outputs'], 1), dtype = float)
			for inputvector, outputvector in self.validationset:
				self.error = configure.functions['add'](self.error, self.criterion(self.net.feedforward(inputvector), outputvector))
		self.error = configure.functions['divide'](self.error, len(self.validationset))
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
				self.error[0][0] -= outputvector[configure.functions['argmax'](self.net.feedforward(inputvector))][0]
		else:
			self.error = numpy.zeros((self.net.dimensions['outputs'], 1), dtype = float)
			for inputvector, outputvector in self.testingset:
				self.error = configure.functions['add'](self.error, self.criterion(self.net.feedforward(inputvector), outputvector))
		self.error = configure.functions['divide'](self.error, len(self.testingset))
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
		self.hypercriterion = hypercriterion if hypercriterion is not None else configure.functions['sum'] # default set to sum

	def gridsearch(self, hyperparameters, batch = 1, iterations = 1, classification = None):
		'''
			Method to optimize hyperparameters by Grid Search
			: param hyperparameters : initial values of hyperparameters to be optimized. eg. [('applyvelocity', [.3, .5]), ('applylearningrate', [.025, .05])]
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param classification : parameter to control whether task is classification
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
			error = self.hypercriterion(self.validate(classification))
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

	def NelderMead(self, hyperparameters, batch = 1, iterations = 1, classification = None, alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5, threshold = 0.05, hyperiterations = 10): # defaults set
		'''
			Method to optimize hyperparameters by Nelder Mead Algorithm
			: param hyperparameters : initial values of hyperparameters to be optimized. eg. [('applyvelocity', 'applylearningrate'), [.5, .025], [.3, .05], [.1, .025]]
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param classification : parameter to control whether task is classification
			: param alpha : Nelder Mead Algorithm reflection parameter
			: param gamma : Nelder Mead Algorithm expansion parameter
			: param rho : Nelder Mead Algorithm contraction parameter
			: param sigma : Nelder Mead Algorithm reduction parameter
			: param threshold : distance from centroid threshold for termination
			: param hyperiterations : hyperiteration threshold for termination
		'''
		def geterror(self, dimensions, hyperparameters, point, batch, iterations, classification):
			for i in range(dimensions):
				getattr(self.net, hyperparameters[0][i])(point[i])
			self.train(batch, iterations)
			return self.hypercriterion(self.validate(classification))

		backupnet = copy.deepcopy(self.net)
		dimensions = len(hyperparameters[0])
		simplex = [numpy.reshape(hyperparameters[i], (dimensions)) for i in range(1, len(hyperparameters))]
		costs = list()
		besterror = float('inf')
		bestnet = copy.deepcopy(self.net)

		for point in simplex:
			error = geterror(self, dimensions, hyperparameters, point, batch, iterations, classification)
			if error < besterror:
				besterror = error
				bestnet = copy.deepcopy(self.net)
			costs.append(error)
			self.net = copy.deepcopy(backupnet)

		for iteration in range(hyperiterations):
			costs, simplex = zip(*sorted(zip(costs, simplex), key = lambda x: x[0]))
			costs, simplex = list(costs), list(simplex)

			centroid = configure.functions['divide'](configure.functions['sum'](simplex, axis = 0), dimensions)
			if max(configure.functions['norm'](configure.functions['subtract'](centroid, point)) for point in simplex) < threshold:
				break

			reflectedpoint = configure.functions['add'](centroid, configure.functions['multiply'](alpha, configure.functions['subtract'](centroid, simplex[-1])))
			reflectederror = geterror(self, dimensions, hyperparameters, reflectedpoint, batch, iterations, classification)
			if reflectederror < besterror:
				besterror = reflectederror
				bestnet = copy.deepcopy(self.net)
			self.net = copy.deepcopy(backupnet)

			if costs[0] <= reflectederror < costs[-2]:
				simplex[-1] = reflectedpoint
				costs[-1] = reflectederror

			elif reflectederror < costs[0]:
				expandedpoint = configure.functions['add'](centroid, configure.functions['multiply'](gamma, configure.functions['subtract'](centroid, simplex[-1])))
				expandederror = geterror(self, dimensions, hyperparameters, expandedpoint, batch, iterations, classification)
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
					contractedpoint = configure.functions['add'](centroid, configure.functions['multiply'](rho, configure.functions['subtract'](centroid, simplex[-1])))
				else:
					contractedpoint = configure.functions['add'](centroid, configure.functions['multiply'](rho, configure.functions['subtract'](simplex[-1], centroid)))
				contractederror = geterror(self, dimensions, hyperparameters, contractedpoint, batch, iterations, classification)

				if contractederror < besterror:
					besterror = contractederror
					bestnet = copy.deepcopy(self.net)
				self.net = copy.deepcopy(backupnet)

				if contractederror < costs[-1]:
					simplex[-1] = contractedpoint
					costs[-1] = contractederror
				else:
					for i in range(1, len(simplex)):
						simplex[i] = configure.functions['add'](simplex[0], configure.functions['multiply'](sigma, configure.functions['subtract'](simplex[i], simplex[0])))
						costs[i] = geterror(self, dimensions, hyperparameters, simplex[i], batch, iterations, classification)
						if costs[i] < besterror:
							besterror = costs[i]
							bestnet = copy.deepcopy(self.net)
						self.net = copy.deepcopy(backupnet)

		self.net = copy.deepcopy(bestnet)
		return [(cost, point.tolist()) for cost, point in zip(costs, simplex)]

def distributedcomputation(serialnet, netarrays, trainingset, testingset, validationset = None, criterion = None, batch = 1, iterations = 1, classification = None):
	'''
		Constructor
		: param serialnet : serialized net whose parameters are to be optimized
		: param netarrays : list of numpy arrays in network
		: param trainingset : tuple of absolute path to directory and file containing serialized supervised data set used for training
		: param testingset : tuple of absolute path to directory and file containing serialized supervised data set used for testing
		: param validationset : tuple of absolute path to directory and file containing serialized supervised data set used for validation
		: param criterion : class name of error criterion
		: param batch : training minibatch size
		: param iterations : iteration threshold for termination
		: param classification : parameter to control whether task is classification
	'''
	import net, data
	myoptimizer = net.DistributedOptimizer(serialnet, netarrays, trainingset, testingset, validationset, criterion)
	myoptimizer.train(batch, iterations)
	classify, classifyarray = data.models.serialize(myoptimizer.validate(classification))
	model, modelarray = data.models.serialize(myoptimizer.net)
	return classify, classifyarray, model, modelarray

class DistributedOptimizer(Optimizer):
	'''
		Distributed Optimizer Class
	'''
	def __init__(self, serialnet, netarrays, trainingset, testingset, validationset = None, criterion = None):
		'''
			Constructor
			: param serialnet : serialized net whose parameters are to be optimized
			: param netarrays : list of numpy arrays in network
			: param trainingset : tuple of absolute path to directory and file containing serialized supervised data set used for training
			: param testingset : tuple of absolute path to directory and file containing serialized supervised data set used for testing
			: param validationset : tuple of absolute path to file containing serialized supervised data set used for validation
			: param criterion : class name of error criterion
		'''
		import net
		network = data.models.deserialize(serialnet, netarrays)
		trainingset = data.models.load(*trainingset)
		testingset = data.models.load(*testingset)
		validationset = data.models.load(*validationset) if validationset is not None else self.testingset # default set to testing set
		criterion = getattr(getattr(net, criterion), 'compute') if criterion is not None else criterion
		Optimizer.__init__(self, network, trainingset, testingset, validationset, criterion)

class DistributedHyperoptimizer(base.Net):
	'''
		Distributed Hyperoptimizer Class
	'''
	def __init__(self, net, trainingset, testingset, validationset = None, criterion = None, hypercriterion = None):
		'''
			Constructor
			: param net : net whose parameters are to be optimized
			: param trainingset : tuple of absolute path to directory and file containing serialized supervised data set used for training
			: param testingset : tuple of absolute path to directory and file containing serialized supervised data set used for testing
			: param validationset : tuple of absolute path to directory and file containing serialized supervised data set used for validation
			: param criterion : class name of error criterion
			: param hypercriterion : criterion used to scalarize vectorized error
		'''
		self.net = net
		self.serialnet = data.models.serialize(self.net)
		self.trainingset = trainingset
		self.testingset = testingset
		self.validationset = validationset if validationset is not None else self.testingset # default set to testing set
		self.criterion = criterion if criterion is not None else 'MeanSquared' # default set to half mean squared
		self.hypercriterion = hypercriterion if hypercriterion is not None else configure.functions['sum'] # default set to sum

	def gridsearch(self, hyperparameters, batch = 1, iterations = 1, classification = None):
		'''
			Method to optimize hyperparameters by Grid Search
			: param hyperparameters : initial values of hyperparameters to be optimized. eg. [('applyvelocity', [.3, .5]), ('applylearningrate', [.025, .05])]
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param classification : parameter to control whether task is classification
		'''
		indices = [0 for i in range(len(hyperparameters))]
		bestindices = [0 for i in range(len(hyperparameters))]
		limits = [len(hyperparameters[i][1]) for i in range(len(hyperparameters))]
		besterror = float('inf')
		bestnet = data.models.serialize(self.net)

		cluster = dispy.JobCluster(distributedcomputation)
		jobs = list()

		while not(indices[len(hyperparameters) - 1] == limits[len(hyperparameters) - 1]):
			optimizenet = copy.deepcopy(self.net)
			for i in range(len(hyperparameters)):
				getattr(optimizenet, hyperparameters[i][0])(hyperparameters[i][1][indices[i]])
			model, arrays = data.models.serialize(optimizenet)
			job = cluster.submit(model, arrays, self.trainingset, self.testingset, self.validationset, self.criterion, batch, iterations, classification)
			job.id = copy.deepcopy(indices)
			jobs.append(job)

			indices[0] += 1
			for i in range(len(indices) - 1):
				if indices[i] == limits[i]:
					indices[i + 1] += 1
					indices[i] = 0
				else:
					break

		for job in jobs:
			serialvalidation, serialvalidationarrays, serialnet, serialnetarrays = job()
			error = self.hypercriterion(data.models.deserialize(serialvalidation, serialvalidationarrays))
			if error < besterror:
				besterror = error
				bestindices = job.id
				bestnet = serialnet
				bestnetarrays = serialnetarrays

		self.net = data.models.deserialize(bestnet, bestnetarrays)
		return [hyperparameters[i][1][bestindices[i]] for i in range(len(hyperparameters))], besterror

	def NelderMead(self, hyperparameters, batch = 1, iterations = 1, classification = None, alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5, threshold = 0.05, hyperiterations = 10): # defaults set
		'''
			Method to optimize hyperparameters by Nelder Mead Algorithm
			: param hyperparameters : initial values of hyperparameters to be optimized. eg. [('applyvelocity', 'applylearningrate'), [.5, .025], [.3, .05], [.1, .025]]
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param classification : parameter to control whether task is classification
			: param alpha : Nelder Mead Algorithm reflection parameter
			: param gamma : Nelder Mead Algorithm expansion parameter
			: param rho : Nelder Mead Algorithm contraction parameter
			: param sigma : Nelder Mead Algorithm reduction parameter
			: param threshold : distance from centroid threshold for termination
			: param hyperiterations : hyperiteration threshold for termination
		'''
		def submittocluster(self, cluster, jobs, dimensions, hyperparameters, point, batch, iterations, classification):
			optimizenet = copy.deepcopy(self.net)
			for i in range(dimensions):
				getattr(optimizenet, hyperparameters[0][i])(point[i])
			model, arrays = data.models.serialize(optimizenet)
			job = cluster.submit(model, arrays, self.trainingset, self.testingset, self.validationset, self.criterion, batch, iterations, classification)
			job.id = point
			jobs.append(job)

		backupnet = copy.deepcopy(self.net)
		dimensions = len(hyperparameters[0])
		simplex = [numpy.reshape(hyperparameters[i], (dimensions)) for i in range(1, len(hyperparameters))]
		costs = list()
		besterror = float('inf')
		bestnet = data.models.serialize(self.net)

		cluster = dispy.JobCluster(distributedcomputation)
		jobs = list()

		for point in simplex:
			submittocluster(self, cluster, jobs, dimensions, hyperparameters, point, batch, iterations, classification)

		for job in jobs:
			serialvalidation, serialvalidationarrays, serialnet, serialnetarrays = job()
			error = self.hypercriterion(data.models.deserialize(serialvalidation, serialvalidationarrays))
			if error < besterror:
				besterror = error
				bestindices = job.id
				bestnet = serialnet
				bestnetarrays = serialnetarrays
			costs.append(error)

		for iteration in range(hyperiterations):
			costs, simplex = zip(*sorted(zip(costs, simplex), key = lambda x: x[0]))
			costs, simplex = list(costs), list(simplex)

			centroid = configure.functions['divide'](configure.functions['sum'](simplex, axis = 0), dimensions)
			if max(configure.functions['norm'](configure.functions['subtract'](centroid, point)) for point in simplex) < threshold:
				break

			jobs = list()

			reflectedpoint = configure.functions['add'](centroid, configure.functions['multiply'](alpha, configure.functions['subtract'](centroid, simplex[-1])))
			submittocluster(self, cluster, jobs, dimensions, hyperparameters, reflectedpoint, batch, iterations, classification)
			expandedpoint = configure.functions['add'](centroid, configure.functions['multiply'](gamma, configure.functions['subtract'](centroid, simplex[-1])))
			submittocluster(self, cluster, jobs, dimensions, hyperparameters, expandedpoint, batch, iterations, classification)
			contractedpoint1 = configure.functions['add'](centroid, configure.functions['multiply'](rho, configure.functions['subtract'](centroid, simplex[-1])))
			submittocluster(self, cluster, jobs, dimensions, hyperparameters, contractedpoint1, batch, iterations, classification)
			contractedpoint2 = configure.functions['add'](centroid, configure.functions['multiply'](rho, configure.functions['subtract'](simplex[-1], centroid)))
			submittocluster(self, cluster, jobs, dimensions, hyperparameters, contractedpoint2, batch, iterations, classification)

			serialvalidation, serialvalidationarrays, serialreflectednet, serialreflectednetarrays = jobs[0]()
			reflectederror = self.hypercriterion(data.models.deserialize(serialvalidation, serialvalidationarrays))
			serialvalidation, serialvalidationarrays, serialexpandednet, serialexpandednetarrays = jobs[1]()
			expandederror = self.hypercriterion(data.models.deserialize(serialvalidation, serialvalidationarrays))
			serialvalidation, serialvalidationarrays, serialcontractednet1, serialcontractednetarrays1 = jobs[2]()
			contractederror1 = self.hypercriterion(data.models.deserialize(serialvalidation, serialvalidationarrays))
			serialvalidation, serialvalidationarrays, serialcontractednet2, serialcontractednetarrays2 = jobs[3]()
			contractederror2 = self.hypercriterion(data.models.deserialize(serialvalidation, serialvalidationarrays))

			if reflectederror < besterror:
				besterror = reflectederror
				bestnet = serialreflectednet
				bestnetarrays = serialreflectednetarrays

			if costs[0] <= reflectederror < costs[-2]:
				simplex[-1] = reflectedpoint
				costs[-1] = reflectederror

			elif reflectederror < costs[0]:
				if expandederror < besterror:
					besterror = expandederror
					bestnet = serialexpandednet
					bestnetarrays = serialexpandednetarrays

				if expandederror < reflectederror:
					simplex[-1] = expandedpoint
					costs[-1] = expandederror
				else:
					simplex[-1] = reflectedpoint
					costs[-1] = reflectederror

			else:
				if reflectederror < costs[-1]:
					contractederror = contractederror1
					contractedpoint = contractedpoint1
					if contractederror < besterror:
						besterror = contractederror
						bestnet = serialcontractednet1
						bestnetarrays = serialcontractednetarrays1
				else:
					contractederror = contractederror2
					contractedpoint = contractedpoint2
					if contractederror < besterror:
						besterror = contractederror
						bestnet = serialcontractednet2
						bestnetarrays = serialcontractednetarrays2

				if contractederror < costs[-1]:
					simplex[-1] = contractedpoint
					costs[-1] = contractederror

				else:
					jobs = list()
					for i in range(1, len(simplex)):
						simplex[i] = configure.functions['add'](simplex[0], configure.functions['multiply'](sigma, configure.functions['subtract'](simplex[i], simplex[0])))
						submittocluster(self, cluster, jobs, dimensions, hyperparameters, simplex[i], batch, iterations, classification)

					for i in range(len(jobs)):
						serialvalidation, serialvalidationarrays, serialnet, serialnetarrays = job()
						costs[i + 1] = self.hypercriterion(data.models.deserialize(serialvalidation, serialvalidationarrays))
						if costs[i + 1] < besterror:
							besterror = costs[i + 1]
							bestnet = serialnet
							bestnetarrays = serialnetarrays

		self.net = data.models.deserialize(bestnet, bestnetarrays)
		return [(cost, point.tolist()) for cost, point in zip(costs, simplex)]
