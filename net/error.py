import numpy

class Error:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	function = None
	derivative = None

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.derivative(self.previousoutput, outputvector)
		
class MeanSquared(Error):

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: x)
		self.derivative = numpy.subtract

class CrossEntropy(Error):

	epsilon = None

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.epsilon = 0.0001
		self.function = numpy.vectorize(lambda x: x)
		self.derivative = numpy.vectorize(lambda x, y: (1.0 - y) / (1.0 - x + self.epsilon) - y / (x + self.epsilon))

class NegativeLogLikelihood(Error):

	epsilon = None

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.epsilon = 0.0001
		self.function = numpy.vectorize(lambda x: x)
		self.derivative = numpy.vectorize(lambda x, y: - y / (x + self.epsilon))

class CrossSigmoid(Error):

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 1.0 / (1.0 + numpy.exp(-x)))
		self.derivative = numpy.subtract

class LogSoftMax(Error):

	def __init__(self, inputs):
		Error.__init__(self, inputs)
		self.derivative = numpy.subtract

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		inputvector = numpy.subtract(inputvector, numpy.amax(inputvector))
		inputvector = numpy.exp(inputvector)
		self.previousoutput = numpy.divide(inputvector, numpy.sum(inputvector))
		return self.previousoutput
