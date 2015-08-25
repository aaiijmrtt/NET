import numpy

class Linear:

	inputs = None
	outputs = None
	updates = None

	weights = None
	biases = None

	alpha = None
	eta = None
	learningrate = None

	deltaweights = None
	deltabiases = None

	velocity = None
	regularization = None
	dropout = None

	previousinput = None
	previousoutput = None

	def __init__(self, inputs, outputs, alpha = None, eta = None):
		self.inputs = inputs
		self.outputs = outputs
		self.weights = numpy.random.normal(0.0, 1.0, (self.outputs, self.inputs))
		self.biases = numpy.random.normal(0.0, 1.0, (self.outputs, 1))
		self.applylearningrate(alpha)
		self.applydecayrate(eta)
		self.updates = 0
		self.velocity = None
		self.regularization = None
		self.dropout = None
		self.cleardeltas()

	def cleardeltas(self):
		if self.regularization is not None:
			self.deltaweights, self.deltabiases = self.regularization.cleardeltas()
		else:
			self.deltaweights = numpy.zeros((self.outputs, self.inputs), dtype = float)
			self.deltabiases = numpy.zeros((self.outputs, 1), dtype = float)

	def updateweights(self):
		if self.velocity is not None:
			self.deltaweights, self.deltabiases = self.velocity.updateweights()
		else:
			self.deltaweights = numpy.multiply(self.alpha / (1.0 + self.updates * self.eta), self.deltaweights)
			self.deltabiases = numpy.multiply(self.alpha / (1.0 + self.updates * self.eta), self.deltabiases)
		self.weights = numpy.subtract(self.weights, self.deltaweights)
		self.biases = numpy.subtract(self.biases, self.deltabiases)
		self.cleardeltas()
		self.updates += 1

	def feedforward(self, inputvector):
		if self.dropout is not None:
			self.previousinput = self.dropout.feedforward(inputvector)
		else:
			self.previousinput = inputvector
		self.previousoutput = numpy.add(numpy.dot(self.weights, self.previousinput), self.biases)
		return self.previousoutput

	def backpropagate(self, outputvector):
		self.deltaweights = numpy.add(self.deltaweights, numpy.dot(outputvector, self.previousinput.transpose()))
		self.deltabiases = numpy.add(self.deltabiases, outputvector)
		return numpy.dot(self.weights.transpose(), outputvector)

	def trainingsetup(self):
		self.updates = 0
		if self.velocity is not None:
			self.velocity.velocityweights = numpy.zeros(self.weights.shape, dtype = float)
			self.velocity.velocitybiases = numpy.zeros(self.biases.shape, dtype = float)
		if self.dropout is not None:
			self.dropout.dropout = True

	def testingsetup(self):
		if self.velocity is not None:
			self.velocity.velocityweights = None
			self.velocity.velocitybiases = None
		if self.dropout is not None:
			self.dropout.dropout = False

	def applylearningrate(self, alpha = None):
		self.alpha = alpha if alpha is not None else 0.05 # default set at 0.05

	def applydecayrate(self, eta = None):
		self.eta = eta if eta is not None else 0.0 # default set at 0.0

	def applyvelocity(self, gamma = None):
		self.velocity = Velocity(self, gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		self.regularizer = Regularization(self, lamda, regularizer)

	def applydropout(self, rho = None):
		self.dropout = Dropout(self, rho)

class Velocity:

	linear = None
	gamma = None

	velocityweights = None
	velocitybiases = None

	def __init__(self, linear, gamma = None):
		self.linear = linear
		self.velocityweights = numpy.zeros(self.linear.weights.shape, dtype = float)
		self.velocitybiases = numpy.zeros(self.linear.biases.shape, dtype = float)
		self.gamma = gamma if gamma is not None else 0.5 # default set at 0.5

	def updateweights(self):
		self.velocityweights = numpy.add(numpy.multiply(self.gamma, self.velocityweights), numpy.multiply(self.linear.alpha / (1.0 + self.linear.updates * self.linear.eta), self.linear.deltaweights))
		self.velocitybiases = numpy.add(numpy.multiply(self.gamma, self.velocitybiases), numpy.multiply(self.linear.alpha / (1.0 + self.linear.updates * self.linear.eta), self.linear.deltabiases))
		return self.velocityweights, self.velocitybiases

class Regularization:

	linear = None
	lamda = None
	regularizer = None

	def __init__(self, linear, lamda = None, regularizer = None):
		self.linear = linear
		self.lamda = lamda if lamda is not None else 0.05 # default set at 0.05
		self.regularizer = regularizer if regularizer is not None else numpy.vectorize(lambda x: x) # default set to L2 regularization

	def cleardeltas(self):
		return numpy.multiply(self.lamda, self.regularizer(self.linear.weights)), numpy.multiply(self.lamda, self.regularizer(self.linear.biases))

class Dropout:

	linear = None
	dropout = None
	rho = None
	hadamard = None

	def __init__(self, linear, rho = None):
		self.linear = linear
		self.rho = rho if rho is not None else 0.75 # default set at 0.75
		self.hadamard = numpy.vectorize(lambda x, y: x * y)
		self.dropout = True

	def feedforward(self, inputvector):
		if self.dropout:
			return self.hadamard(numpy.random.binomial(1, self.rho, (self.linear.inputs, 1)), inputvector)
		else:
			return numpy.multiply(self.rho, inputvector)

class Split:

	inputs = None
	outputs = None

	parameter = None

	previousinput = None
	previousoutput = None

	def __init__(self, inputs, parameter):
		self.inputs = inputs
		self.parameter = parameter
		self.outputs = self.inputs * self.parameter

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = numpy.concatenate([self.previousinput] * self.parameter)
		return self.previousoutput

	def backpropagate(self, outputvector):
		deltas = numpy.zeros((self.inputs, 1), dtype = float)
		for i in range(self.outputs):
			deltas[i % self.inputs][0] += outputvector[i][0]
		return deltas

class MergeSum:

	inputs = None
	outputs = None

	parameter = None

	previousinput = None
	previousoutput = None

	def __init__(self, outputs, parameter):
		self.outputs = outputs
		self.parameter = parameter
		self.inputs = self.outputs * self.parameter

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = numpy.zeros((self.outputs, 1), dtype = float)
		for i in range(self.inputs):
			self.previousoutput[i % self.outputs][0] += self.previousinput[i][0]
		return self.previousoutput

	def backpropagate(self, outputvector):
		return numpy.concatenate([outputvector] * self.parameter)

class MergeProduct:

	inputs = None
	outputs = None

	parameter = None

	previousinput = None
	previousoutput = None

	def __init__(self, outputs, parameter):
		self.outputs = outputs
		self.parameter = parameter
		self.inputs = self.outputs * self.parameter

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = numpy.ones((self.outputs, 1), dtype = float)
		for i in range(self.inputs):
			self.previousoutput[i % self.outputs][0] *= self.previousinput[i][0]
		return self.previousoutput

	def backpropagate(self, outputvector):
		deltas = numpy.concatenate([outputvector] * self.parameter)
		for i in range(self.inputs):
			deltas[i][0] *= self.previousoutput[i % self.outputs][0] / self.previousinput[i][0]
		return deltas

class Step:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.previousinput
		return self.previousoutput

	def backpropagate(self, outputvector):
		return outputvector
