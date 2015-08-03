import numpy

class Linear:

	inputs = None
	outputs = None

	weights = None
	biases = None

	alpha = None
	gamma = None

	deltaweights = None
	deltabiases = None

	velocityweights = None
	velocitybiases = None

	previousinput = None
	previousoutput = None

	def __init__(self, inputs, outputs, alpha = None, gamma = None):
		self.inputs = inputs
		self.outputs = outputs
		self.weights = numpy.random.normal(0.0, 1.0, (self.outputs, self.inputs))
		self.biases = numpy.random.normal(0.0, 1.0, (self.outputs, 1))
		self.deltaweights = numpy.zeros((self.outputs, self.inputs), dtype = float)
		self.deltabiases = numpy.zeros((self.outputs, 1), dtype = float)
		self.velocityweights = numpy.zeros((self.outputs, self.inputs), dtype = float)
		self.velocitybiases = numpy.zeros((self.outputs, 1), dtype = float)
		self.alpha = alpha if alpha is not None else 0.05 # default set at 0.05
		self.gamma = gamma if gamma is not None else 0.0 # default set at 0.0

	def cleardeltas(self):
		self.deltaweights = numpy.zeros((self.outputs, self.inputs), dtype = float)
		self.deltabiases = numpy.zeros((self.outputs, 1), dtype = float)

	def updateweights(self):
		self.velocityweights = numpy.add(numpy.multiply(self.gamma, self.velocityweights), numpy.multiply(self.alpha, self.deltaweights))
		self.velocitybiases = numpy.add(numpy.multiply(self.gamma, self.velocitybiases), numpy.multiply(self.alpha, self.deltabiases))
		self.weights = numpy.subtract(self.weights, self.velocityweights)
		self.biases = numpy.subtract(self.biases, self.velocitybiases)
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = numpy.add(numpy.dot(self.weights, self.previousinput), self.biases)
		return self.previousoutput

	def backpropagate(self, outputvector):
		self.deltaweights = numpy.add(self.deltaweights, numpy.dot(outputvector, self.previousinput.transpose()))
		self.deltabiases = numpy.add(self.deltabiases, outputvector)
		return numpy.dot(self.weights.transpose(), outputvector)
