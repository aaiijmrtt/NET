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

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		self.modifier.applyadaptivegain(tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		self.modifier.applyrootmeansquarepropagation(meu)

	def applyadaptivegradient(self):
		self.modifier.applyadaptivegradient()

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
		self.previousoutput = numpy.add(numpy.multiply(self.weights, self.previousnormalized), self.biases)
		return self.previousoutput

	def backpropagate(self, outputvector): # ignores dropout
		self.deltaweights = numpy.add(self.deltaweights, numpy.multiply(outputvector, self.previousnormalized))
		self.deltabiases = numpy.add(self.deltabiases, outputvector)
		return numpy.multiply(numpy.divide(self.weights, self.batch), numpy.divide(numpy.subtract(self.batch - 1, numpy.square(self.previousnormalized)), numpy.sqrt(numpy.add(self.epsilon, self.variance))))

	def normalize(self):
		self.mean = numpy.divide(self.linearsum, self.batch)
		self.variance = numpy.subtract(numpy.divide(self.quadraticsum, self.batch), numpy.square(self.mean))

	def accumulatingsetup(self):
		self.batch = 0
		self.linearsum = numpy.zeros((self.inputs, 1), dtype = float)
		self.quadraticsum = numpy.zeros((self.inputs, 1), dtype = float)

class Modifier:

	epsilon = 0.0001

	L1regularizer = numpy.vectorize(lambda x: 1.0 if x > 0.0 else -1.0 if x < 0.0 else 0.0)
	L2regularizer = numpy.vectorize(lambda x: x)

	def __init__(self, layer):
		self.layer = layer

		self.velocity = False
		self.gamma = None
		self.velocityweights = None
		self.velocitybiases = None

		self.dropout = False
		self.rho = None

		self.lamda = None
		self.regularizer = None
		self.regularization = False

		self.adaptivegain = False
		self.tau = None
		self.maxgain = None
		self.mingain = None
		self.gainadapter = None
		self.gainclipper = None
		self.gainweights = None
		self.gainbiases = None
		self.olddeltaweights = None
		self.olddeltabiases = None

		self.rootmeansquarepropagation = False
		self.meu = None
		self.meansquareweights = None
		self.meansquarebiases = None

		self.adaptivegradient = False
		self.sumsquareweights = None
		self.sumsquarebiases = None

		self.training = None

	def applyvelocity(self, gamma = None):
		self.velocity = True
		self.velocityweights = numpy.zeros(self.layer.weights.shape, dtype = float)
		self.velocitybiases = numpy.zeros(self.layer.biases.shape, dtype = float)
		self.gamma = gamma if gamma is not None else 0.5 # default set at 0.5

	def applyregularization(self, lamda = None, regularizer = None):
		self.regularization = True
		self.lamda = lamda if lamda is not None else 0.005 # default set at 0.005
		self.regularizer = regularizer if regularizer is not None else Modifier.L2regularizer # default set to L2 regularization

	def applydropout(self, rho = None):
		self.dropout = True
		self.training = False
		self.rho = rho if rho is not None else 0.75 # default set at 0.75

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		self.adaptivegain = True
		self.tau = tau if tau is not None else 0.05 # default set to 0.05
		self.maxgain = maximum if maximum is not None else 100.0 # default set to 100.0
		self.mingain = minimum if minimum is not None else 0.01 # default set to 0.01
		self.gainadapter = numpy.vectorize(lambda x, y, z: z + self.tau if x * y > 0.0 else z * (1.0 - self.tau))
		self.gainclipper = numpy.vectorize(lambda x: self.mingain if x < self.mingain else self.maxgain if self.maxgain < x else x)
		self.gainweights = numpy.ones(self.layer.weights.shape, dtype = float)
		self.gainbiases = numpy.ones(self.layer.biases.shape, dtype = float)
		self.olddeltaweights = numpy.copy(self.layer.deltaweights)
		self.olddeltabiases = numpy.copy(self.layer.deltabiases)

	def applyrootmeansquarepropagation(self, meu = None):
		self.rootmeansquarepropagation = True
		self.meu = meu if meu is not None else 0.9 # default set to 0.9
		self.meansquareweights = numpy.zeros(self.layer.weights.shape, dtype = float)
		self.meansquarebiases = numpy.zeros(self.layer.biases.shape, dtype = float)

	def applyadaptivegradient(self):
		self.adaptivegradient = True
		self.sumsquareweights = numpy.zeros(self.layer.weights.shape, dtype = float)
		self.sumsquarebiases = numpy.zeros(self.layer.biases.shape, dtype = float)

	def updateweights(self):
		deltaweights = numpy.multiply(self.layer.alpha / (1.0 + self.layer.updates * self.layer.eta), self.layer.deltaweights)
		deltabiases = numpy.multiply(self.layer.alpha / (1.0 + self.layer.updates * self.layer.eta), self.layer.deltabiases)

		if self.rootmeansquarepropagation:
			self.meansquareweights = numpy.add(numpy.multiply(self.meu, self.meansquareweights), numpy.multiply(1.0 - self.meu, numpy.square(self.layer.deltaweights)))
			self.meansquarebiases = numpy.add(numpy.multiply(self.meu, self.meansquarebiases), numpy.multiply(1.0 - self.meu, numpy.square(self.layer.deltabiases)))
			deltaweights = numpy.divide(deltaweights, numpy.sqrt(numpy.add(Modifier.epsilon, self.meansquareweights)))
			deltabiases = numpy.divide(deltabiases, numpy.sqrt(numpy.add(Modifier.epsilon, self.meansquarebiases)))

		if self.adaptivegradient:
			self.sumsquareweights = numpy.add(self.sumsquareweights, numpy.square(self.layer.deltaweights))
			self.sumsquarebiases = numpy.add(self.sumsquarebiases, numpy.square(self.layer.deltabiases))
			deltaweights = numpy.divide(deltaweights, numpy.sqrt(numpy.add(Modifier.epsilon, self.sumsquareweights)))
			deltabiases = numpy.divide(deltabiases, numpy.sqrt(numpy.add(Modifier.epsilon, self.sumsquarebiases)))

		if self.adaptivegain:
			self.gainweights = self.gainclipper(self.gainadapter(self.olddeltaweights, self.layer.deltaweights, self.gainweights))
			self.gainbiases = self.gainclipper(self.gainadapter(self.olddeltabiases, self.layer.deltabiases, self.gainbiases))
			self.olddeltaweights = numpy.copy(self.layer.deltaweights)
			self.oldbiases = numpy.copy(self.layer.deltabiases)
			deltaweights = numpy.multiply(self.gainweights, deltaweights)
			deltabiases = numpy.multiply(self.gainbiases, deltabiases)

		if self.velocity:
			self.velocityweights = numpy.add(numpy.multiply(self.gamma, self.velocityweights), deltaweights)
			self.velocitybiases = numpy.add(numpy.multiply(self.gamma, self.velocitybiases), deltabiases)
			deltaweights = self.velocityweights
			deltabiases = self.velocitybiases

		return numpy.copy(deltaweights), numpy.copy(deltabiases)

	def cleardeltas(self):
		if not self.regularization:
			return numpy.zeros(self.layer.weights.shape, dtype = float), numpy.zeros(self.layer.biases.shape, dtype = float)
		return numpy.multiply(self.lamda, self.regularizer(self.layer.weights)), numpy.multiply(self.lamda, self.regularizer(self.layer.biases))

	def feedforward(self, inputvector):
		if not self.dropout:
			return inputvector
		return numpy.multiply(numpy.random.binomial(1, self.rho, inputvector.shape), inputvector) if self.training else numpy.multiply(self.rho, inputvector)

	def trainingsetup(self):
		if self.velocity:
			self.applyvelocity(self.gamma)
		if self.adaptivegain:
			self.applyadaptivegain(self. tau, self.maxgain, self.mingain)
		if self.rootmeansquarepropagation:
			self.applyrootmeansquarepropagation(self.meu)
		if self.adaptivegradient:
			self.applyadaptivegradient()
		self.training = True

	def testingsetup(self):
		if self.velocity:
			self.velocityweights = None
			self.velocitybiases = None
		if self.adaptivegain:
			self.gainadapter = None
			self.gainclipper = None
			self.gainweights = None
			self.gainbiases = None
			self.olddeltaweights = None
			self.olddeltabiases = None
		if self.rootmeansquarepropagation:
			self.meansquareweights = None
			self.meansquarebiases = None
		if self.adaptivegradient:
			self.sumsquareweights = None
			self.sumsquarebiases = None
		self.training = False
