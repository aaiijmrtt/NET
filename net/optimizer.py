import numpy, copy

class Optimizer:

	net = None

	trainingset = None
	validationset = None
	testingset = None

	criterion = None
	error = None

	def __init__(self, net, trainingset, testingset, validationset = None, criterion = None):
		self.net = net
		self.trainingset = trainingset
		self.validationset = validationset if validationset is not None else testingset # default set to testing set
		self.testingset = testingset
		self.criterion = numpy.vectorize(criterion) if criterion is not None else numpy.vectorize(lambda x, y: 0.5 * (x - y) ** 2) # default set to half mean squared

	def train(self, batch = 1, iterations = 1):
		self.net.timingsetup()
		self.net.trainingsetup()
		for i in range(iterations):
			for j in range(len(self.trainingset)):
				if j % batch == 0:
					self.net.updateweights()
				self.net.feedforward(self.trainingset[j][0])
				self.net.backpropagate(self.trainingset[j][1])

	def validate(self):
		self.net.timingsetup()
		self.net.testingsetup()
		self.error = numpy.zeros((self.net.outputs, 1), dtype = float)
		for inputvector, outputvector in self.validationset:
			self.error = numpy.add(self.error, self.criterion(self.net.feedforward(inputvector), outputvector))
		self.error = numpy.divide(self.error, len(self.testingset))
		return self.error

	def test(self):
		self.net.timingsetup()
		self.net.testingsetup()
		self.error = numpy.zeros((self.net.outputs, 1), dtype = float)
		for inputvector, outputvector in self.testingset:
			self.error = numpy.add(self.error, self.criterion(self.net.feedforward(inputvector), outputvector))
		self.error = numpy.divide(self.error, len(self.testingset))
		return self.error

class Hyperoptimizer:

	optimizer = None
	criterion = None

	def __init__(self, optimizer, criterion = None):
		self.optimizer = optimizer
		self.criterion = criterion if criterion is not None else lambda x: numpy.sum(x) / numpy.product(x.shape) # default set to average

	def gridsearch(self, hyperparameters, batch = 1, iterations = 1):
		backupnet = copy.deepcopy(self.optimizer.net)
		indices = [0 for i in range(len(hyperparameters))]
		bestindices = [0 for i in range(len(hyperparameters))]
		limits = [len(hyperparameters[i][1]) for i in range(len(hyperparameters))]
		besterror = float('inf')
		bestnet = backupnet
		while not(indices[len(hyperparameters) - 1] == limits[len(hyperparameters) - 1]):
			for i in range(len(hyperparameters)):
				getattr(self.optimizer.net, hyperparameters[i][0])(hyperparameters[i][1][indices[i]])
			self.optimizer.train(batch, iterations)
			error = self.criterion(self.optimizer.validate())
			if error < besterror:
				besterror = error
				bestindices = copy.deepcopy(indices)
				bestnet = copy.deepcopy(self.optimizer.net)
			indices[0] += 1
			for i in range(len(indices) - 1):
				if indices[i] == limits[i]:
					indices[i + 1] += 1
					indices[i] = 0
				else:
					break
			self.optimizer.net = copy.deepcopy(backupnet)
		self.optimizer.net = copy.deepcopy(bestnet)
		return [hyperparameters[i][1][bestindices[i]] for i in range(len(hyperparameters))]
