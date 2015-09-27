import numpy, net.layer

class Benchmark(net.layer.Layer):

	def __init__(self, weights, biases, multivariatefunction, alpha = None):
		net.layer.Layer.__init__(self, weights, biases, alpha)
		self.multivariatefunction = multivariatefunction
		self.weights = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.inputs, 1))
		self.biases = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.outputs), (self.outputs, 1))
		self.cleardeltas()

	def backpropagate(self, outputvector):
		deltavectors = numpy.split(outputvector, [self.inputs])
		self.deltaweights = numpy.add(self.deltaweights, deltavectors[0])
		self.deltabiases = numpy.add(self.deltabiases, deltavectors[1])

	def train(self, trainingsetsize, batch = 1, iterations = 1):
		self.trainingsetup()
		for i in range(iterations):
			for j in range(trainingsetsize):
				if j % batch == 0:
					self.updateweights()
					self.multivariatefunction.update()
				self.backpropagate(self.multivariatefunction.derivative(numpy.concatenate([self.weights, self.biases])))
		return numpy.subtract(numpy.concatenate([self.weights, self.biases]), self.multivariatefunction.minima())
