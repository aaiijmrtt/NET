import numpy

class Error:

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs
		self.previousinput = None
		self.previousoutput = None
		self.function = None
		self.derivative = None

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.derivative(self.previousoutput, outputvector)

class MeanSquared(Error):

	criterion = numpy.vectorize(lambda x, y: 0.5 * (x - y) ** 2)

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.function = lambda x: x
		self.derivative = numpy.subtract

	@staticmethod
	def compute(inputvector, outputvector):
		return MeanSquared.criterion(inputvector, outputvector)

class CrossEntropy(Error):

	epsilon = 0.0001
	criterion = numpy.vectorize(lambda x, y: - (y * numpy.log(x + CrossEntropy.epsilon) + (1.0 - y) * numpy.log(1.0 - x + CrossEntropy.epsilon)))

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.function = lambda x: x
		self.derivative = numpy.vectorize(lambda x, y: (1.0 - y) / (1.0 - x + CrossEntropy.epsilon) - y / (x + CrossEntropy.epsilon))

	@staticmethod
	def compute(inputvector, outputvector):
		return CrossEntropy.criterion(inputvector, outputvector)

class NegativeLogLikelihood(Error):

	epsilon = 0.0001
	criterion = numpy.vectorize(lambda x, y: - y * numpy.log(x + NegativeLogLikelihood.epsilon))

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.function = lambda x: x
		self.derivative = numpy.vectorize(lambda x, y: - y / (x + NegativeLogLikelihood.epsilon))

	@staticmethod
	def compute(inputvector, outputvector):
		return NegativeLogLikelihood.criterion(inputvector, outputvector)

class CrossSigmoid(Error):

	epsilon = 0.0001
	criterion = numpy.vectorize(lambda x, y: - (y * numpy.log(x + CrossSigmoid.epsilon) + (1.0 - y) * numpy.log(1.0 - x + CrossSigmoid.epsilon)))

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 1.0 / (1.0 + numpy.exp(-x)))
		self.derivative = numpy.subtract

	@staticmethod
	def compute(inputvector, outputvector):
		return CrossSigmoid.criterion(inputvector, outputvector)

class LogSoftMax(Error):

	epsilon = 0.0001
	criterion = numpy.vectorize(lambda x, y: - y * numpy.log(x + LogSoftMax.epsilon))

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.derivative = numpy.subtract

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		inputvector = numpy.subtract(inputvector, numpy.amax(inputvector))
		inputvector = numpy.exp(inputvector)
		self.previousoutput = numpy.divide(inputvector, numpy.sum(inputvector))
		return self.previousoutput

	@staticmethod
	def compute(inputvector, outputvector):
		return LogSoftMax.criterion(inputvector, outputvector)

class KullbackLeiblerDivergence(Error):

	epsilon = 0.0001
	criterion = numpy.vectorize(lambda x, y: y * (numpy.log(y + KullbackLeiblerDivergence.epsilon) - numpy.log(x + KullbackLeiblerDivergence.epsilon)))

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.function = lambda x: x
		self.derivative = numpy.vectorize(lambda x, y: - y / (x + KullbackLeiblerDivergence.epsilon))

	@staticmethod
	def compute(inputvector, outputvector):
		return KullbackLeiblerDivergence.criterion(inputvector, outputvector)

class CosineDistance(Error):

	epsilon = 0.0001

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.function = lambda x: x

	def backpropagate(self, outputvector):
		inputnorm = numpy.sum(numpy.square(self.previousinput))
		outputnorm = numpy.sum(numpy.square(outputvector))
		direction = numpy.sum(numpy.multiply(self.previousinput, outputvector))
		return numpy.divide(numpy.subtract(numpy.multiply(direction, self.previousinput), numpy.multiply(inputnorm, outputvector)), numpy.sqrt(outputnorm) * numpy.sqrt(inputnorm) ** 3)

	@staticmethod
	def compute(inputvector, outputvector):
		inputnorm = numpy.sqrt(numpy.sum(numpy.square(inputvector)))
		outputnorm = numpy.sqrt(numpy.sum(numpy.square(outputvector)))
		direction = numpy.multiply(inputvector, outputvector)
		return - numpy.divide(direction, (inputnorm * outputnorm + CosineDistance.epsilon))
