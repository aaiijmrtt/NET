import numpy

class Optimizer:

	net = None

	trainingset = None
	testingset = None

	criterion = None
	error = None

	def __init__(self, net, trainingset, testingset, criterion):
		self.net = net
		self.trainingset = trainingset
		self.testingset = testingset
		self.criterion = numpy.vectorize(criterion)

	def train(self, iterations = 1):
		self.net.timingsetup()
		self.net.trainingsetup()
		for i in range(iterations):
			for inputvector, outputvector in self.trainingset:
				self.net.feedforward(inputvector)
				self.net.backpropagate(outputvector)
				self.net.updateweights()

	def test(self, iterations = 1):
		self.net.timingsetup()
		self.net.testingsetup()
		self.error = numpy.zeros((self.net.outputs, 1), dtype = float)
		for i in range(iterations):
			for inputvector, outputvector in self.testingset:
				self.error = numpy.add(self.error, self.criterion(self.net.feedforward(inputvector), outputvector))
		self.error = numpy.divide(self.error, len(self.testingset))
		return self.error
