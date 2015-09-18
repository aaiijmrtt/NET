import numpy

class Layer:

	def __init__(self, inputs, outputs, alpha = None, eta = None):
		self.inputs = inputs
		self.outputs = outputs
		self.applylearningrate(alpha)
		self.applydecayrate(eta)
		self.updates = 0
		self.modifier = Modifier(self)
		self.weights = None
		self.biases = None
		self.deltaweights = None
		self.deltabiases = None
		self.previousinput = None
		self.previousoutput = None

	def cleardeltas(self):
		self.deltaweights, self.deltabiases = self.modifier.cleardeltas()

	def updateweights(self):
		self.deltaweights, self.deltabiases = self.modifier.updateweights()
		self.weights = numpy.subtract(self.weights, self.deltaweights)
		self.biases = numpy.subtract(self.biases, self.deltabiases)
		self.cleardeltas()
		self.updates += 1

	def trainingsetup(self):
		self.updates = 0
		self.modifier.trainingsetup()

	def testingsetup(self):
		self.modifier.testingsetup()

	def applylearningrate(self, alpha = None):
		self.alpha = alpha if alpha is not None else 0.05 # default set at 0.05

	def applydecayrate(self, eta = None):
		self.eta = eta if eta is not None else 0.0 # default set at 0.0

	def applyvelocity(self, gamma = None):
		self.modifier.applyvelocity(gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		self.modifier.applyregularization(lamda, regularizer)

	def applydropout(self, rho = None):
		self.modifier.applydropout(rho)

class Linear(Layer):

	def __init__(self, inputs, outputs, alpha = None, eta = None):
		Layer.__init__(self, inputs, outputs, alpha, eta)
		self.weights = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
		self.biases = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, 1))
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previousoutput = numpy.add(numpy.dot(self.weights, self.previousinput), self.biases)
		return self.previousoutput

	def backpropagate(self, outputvector):
		self.deltaweights = numpy.add(self.deltaweights, numpy.dot(outputvector, self.previousinput.transpose()))
		self.deltabiases = numpy.add(self.deltabiases, outputvector)
		return numpy.dot(self.weights.transpose(), outputvector)

class Normalizer(Layer):

	hadamard = numpy.vectorize(lambda x, y: x * y)

	def __init__(self, inputs, alpha = None, eta = None, epsilon = None):
		Layer.__init__(self, inputs, inputs, alpha, eta)
		self.weights = numpy.ones((self.inputs, 1), dtype = float)
		self.biases = numpy.zeros((self.inputs, 1), dtype = float)
		self.mean = numpy.zeros((self.inputs, 1), dtype = float)
		self.variance = numpy.ones((self.inputs, 1), dtype = float)
		self.epsilon = epsilon if epsilon is not None else 0.0001
		self.batch = 1
		self.cleardeltas()
		self.linearsum = None
		self.quadraticsum = None

	def accumulate(self, inputvector):
		self.linearsum = numpy.add(self.linearsum, inputvector)
		self.quadraticsum = numpy.add(self.quadraticsum, numpy.square(inputvector))
		self.batch += 1

	def feedforward(self, inputvector): # ignores dropout
		self.previousinput = inputvector
		self.previousnormalized = numpy.divide(numpy.subtract(self.previousinput, self.mean), numpy.sqrt(numpy.add(self.epsilon, self.variance)))
		self.previousoutput = numpy.add(Normalizer.hadamard(self.weights, self.previousnormalized), self.biases)
		return self.previousoutput

	def backpropagate(self, outputvector): # ignores dropout
		self.deltaweights = numpy.add(self.deltaweights, Normalizer.hadamard(outputvector, self.previousnormalized))
		self.deltabiases = numpy.add(self.deltabiases, outputvector)
		return Normalizer.hadamard(numpy.divide(self.weights, self.batch), numpy.divide(numpy.subtract(self.batch - 1, numpy.square(self.previousnormalized)), numpy.sqrt(numpy.add(self.epsilon, self.variance))))

	def normalize(self):
		self.mean = numpy.divide(self.linearsum, self.batch)
		self.variance = numpy.subtract(numpy.divide(self.quadraticsum, self.batch), numpy.square(self.mean))

	def accumulatingsetup(self):
		self.batch = 0
		self.linearsum = numpy.zeros((self.inputs, 1), dtype = float)
		self.quadraticsum = numpy.zeros((self.inputs, 1), dtype = float)

class Modifier:

	hadamard = numpy.vectorize(lambda x, y: x * y)

	def __init__(self, layer):
		self.layer = layer
		self.gamma = None
		self.lamda = None
		self.rho = None
		self.velocityweights = None
		self.velocitybiases = None
		self.regularizer = None
		self.velocity = False
		self.regularization = False
		self.dropout = False
		self.training = None

	def applyvelocity(self, gamma = None):
		self.velocity = True
		self.velocityweights = numpy.zeros(self.layer.weights.shape, dtype = float)
		self.velocitybiases = numpy.zeros(self.layer.biases.shape, dtype = float)
		self.gamma = gamma if gamma is not None else 0.5 # default set at 0.5

	def applyregularization(self, lamda = None, regularizer = None):
		self.regularization = True
		self.lamda = lamda if lamda is not None else 0.005 # default set at 0.005
		self.regularizer = regularizer if regularizer is not None else numpy.vectorize(lambda x: x) # default set to L2 regularization

	def applydropout(self, rho = None):
		self.dropout = True
		self.training = False
		self.rho = rho if rho is not None else 0.75 # default set at 0.75

	def updateweights(self):
		if not self.velocity:
			return numpy.multiply(self.layer.alpha / (1.0 + self.layer.updates * self.layer.eta), self.layer.deltaweights), numpy.multiply(self.layer.alpha / (1.0 + self.layer.updates * self.layer.eta), self.layer.deltabiases)
		self.velocityweights = numpy.add(numpy.multiply(self.gamma, self.velocityweights), numpy.multiply(self.layer.alpha / (1.0 + self.layer.updates * self.layer.eta), self.layer.deltaweights))
		self.velocitybiases = numpy.add(numpy.multiply(self.gamma, self.velocitybiases), numpy.multiply(self.layer.alpha / (1.0 + self.layer.updates * self.layer.eta), self.layer.deltabiases))
		return self.velocityweights, self.velocitybiases

	def cleardeltas(self):
		if not self.regularization:
			return numpy.zeros(self.layer.weights.shape, dtype = float), numpy.zeros(self.layer.biases.shape, dtype = float)
		return numpy.multiply(self.lamda, self.regularizer(self.layer.weights)), numpy.multiply(self.lamda, self.regularizer(self.layer.biases))

	def feedforward(self, inputvector):
		if not self.dropout:
			return inputvector
		return Modifier.hadamard(numpy.random.binomial(1, self.rho, inputvector.shape), inputvector) if self.training else numpy.multiply(self.rho, inputvector)

	def trainingsetup(self):
		if self.velocity:
			self.velocityweights = numpy.zeros(self.layer.weights.shape, dtype = float)
			self.velocitybiases = numpy.zeros(self.layer.biases.shape, dtype = float)
		self.training = True

	def testingsetup(self):
		if self.velocity:
			self.velocityweights = None
			self.velocitybiases = None
		self.training = False
