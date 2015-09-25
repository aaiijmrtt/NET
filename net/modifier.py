import numpy

class Modifier:

	def __init__(self, layer):
		self.layer = layer
		self.alpha = None
		self.decay = None
		self.dropout = None
		self.regularization = None
		self.velocity = None
		self.adaptivegain = None
		self.rootmeansquarepropagation = None
		self.adaptivegradient = None

	def applylearningrate(self, alpha = None):
		self.alpha = alpha if alpha is not None else 0.05 # default set at 0.05

	def applydecayrate(self, eta = None):
		self.decay = Decay(self, eta)

	def applyvelocity(self, gamma = None):
		self.velocity = Velocity(self, gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		self.regularization = Regularization(self, lamda, regularizer)

	def applydropout(self, rho = None):
		self.dropout = Dropout(self, rho)

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		self.adaptivegain = AdaptiveGain(self, tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		self.rootmeansquarepropagation = RootMeanSquarePropagation(self, meu)

	def applyadaptivegradient(self):
		self.adaptivegradient = AdaptiveGradient(self)

	def updateweights(self):
		learningrate = self.alpha
		if self.decay is not None:
			learningrate = self.decay.updateweights(learningrate)
		deltaweights = numpy.multiply(learningrate, self.layer.deltaweights)
		deltabiases = numpy.multiply(learningrate, self.layer.deltabiases)
		if self.rootmeansquarepropagation is not None:
			deltaweights, deltabiases = self.rootmeansquarepropagation.updateweights(deltaweights, deltabiases)
		if self.adaptivegradient is not None:
			deltaweights, deltabiases = self.adaptivegradient.updateweights(deltaweights, deltabiases)
		if self.adaptivegain is not None:
			deltaweights, deltabiases = self.adaptivegain.updateweights(deltaweights, deltabiases)
		if self.velocity is not None:
			deltaweights, deltabiases = self.velocity.updateweights(deltaweights, deltabiases)
		return deltaweights, deltabiases

	def cleardeltas(self):
		deltaweights, deltabiases = numpy.zeros(self.layer.weights.shape, dtype = float), numpy.zeros(self.layer.biases.shape, dtype = float)
		if self.regularization is not None:
			deltaweights, deltabiases = self.regularization.cleardeltas()
		return deltaweights, deltabiases

	def feedforward(self, inputvector):
		if self.dropout is not None:
			inputvector = self.dropout.feedforward(inputvector)
		return inputvector

	def trainingsetup(self):
		for attribute in self.__dict__:
			if attribute in ['decay', 'dropout', 'velocity', 'adaptivegain', 'adaptivegradient', 'rootmeansquarepropagation']:
				if self.__dict__[attribute] is not None:
					self.__dict__[attribute].trainingsetup()

	def testingsetup(self):
		for attribute in self.__dict__:
			if attribute in ['decay', 'dropout', 'velocity', 'adaptivegain', 'adaptivegradient', 'rootmeansquarepropagation']:
				if self.__dict__[attribute] is not None:
					self.__dict__[attribute].testingsetup()

class Decay:

	def __init__(self, modifier, eta = None):
		self.modifier = modifier
		self.eta = eta if eta is not None else 0.05 # default set at 0.05
		self.updates = 0

	def updateweights(self, learningrate):
		learningrate /= (1.0 + self.updates * self.eta)
		self.updates += 1
		return learningrate

	def trainingsetup(self):
		self.__init__(self.modifier)

	def testingsetup(self):
		self.updates = None

class Dropout:

	def __init__(self, modifier, rho = None):
		self.modifier = modifier
		self.rho = rho if rho is not None else 0.75 # default set at 0.75
		self.training = True

	def feedforward(self, inputvector):
		if self.training:
			return numpy.multiply(numpy.random.binomial(1, self.rho, inputvector.shape), inputvector)
		else:
			return numpy.multiply(self.rho, inputvector)

	def trainingsetup(self):
		self.__init__(self.modifier, self.rho)

	def testingsetup(self):
		self.training = False

class Regularization:

	L1regularizer = numpy.vectorize(lambda x: 1.0 if x > 0.0 else -1.0 if x < 0.0 else 0.0)
	L2regularizer = numpy.vectorize(lambda x: x)

	def __init__(self, modifier, lamda = None, regularizer = None):
		self.modifier = modifier
		self.lamda = lamda if lamda is not None else 0.005 # default set at 0.005
		self.regularizer = regularizer if regularizer is not None else Regularization.L2regularizer # default set to L2 regularization

	def cleardeltas(self):
		return numpy.multiply(self.lamda, self.regularizer(self.modifier.layer.weights)), numpy.multiply(self.lamda, self.regularizer(self.modifier.layer.biases))

class Velocity:

	def __init__(self, modifier, gamma = None):
		self.modifier = modifier
		self.velocityweights = numpy.zeros(self.modifier.layer.weights.shape, dtype = float)
		self.velocitybiases = numpy.zeros(self.modifier.layer.biases.shape, dtype = float)
		self.gamma = gamma if gamma is not None else 0.5 # default set at 0.5

	def updateweights(self, deltaweights, deltabiases):
		self.velocityweights = numpy.add(numpy.multiply(self.gamma, self.velocityweights), deltaweights)
		self.velocitybiases = numpy.add(numpy.multiply(self.gamma, self.velocitybiases), deltabiases)
		deltaweights = numpy.copy(self.velocityweights)
		deltabiases = numpy.copy(self.velocitybiases)
		return deltaweights, deltabiases

	def trainingsetup(self):
		self.__init__(self.modifier, self.gamma)

	def testingsetup(self):
		self.velocityweights = None
		self.velocitybiases = None

class AdaptiveGradient:

	epsilon = 0.0001

	def __init__(self, modifier):
		self.modifier = modifier
		self.sumsquareweights = numpy.zeros(self.modifier.layer.weights.shape, dtype = float)
		self.sumsquarebiases = numpy.zeros(self.modifier.layer.biases.shape, dtype = float)

	def updateweights(self, deltaweights, deltabiases):
		self.sumsquareweights = numpy.add(self.sumsquareweights, numpy.square(self.modifier.layer.deltaweights))
		self.sumsquarebiases = numpy.add(self.sumsquarebiases, numpy.square(self.modifier.layer.deltabiases))
		deltaweights = numpy.divide(deltaweights, numpy.sqrt(numpy.add(AdaptiveGradient.epsilon, self.sumsquareweights)))
		deltabiases = numpy.divide(deltabiases, numpy.sqrt(numpy.add(AdaptiveGradient.epsilon, self.sumsquarebiases)))
		return deltaweights, deltabiases

	def trainingsetup(self):
		self.__init__(self.modifier)

	def testingsetup(self):
		self.sumsquareweights = None
		self.sumsquarebiases = None

class AdaptiveGain:

	def __init__(self, modifier, tau = None, maximum = None, minimum = None):
		self.modifier = modifier
		self.tau = tau if tau is not None else 0.05 # default set to 0.05
		self.maximum = maximum if maximum is not None else 100.0 # default set to 100.0
		self.minimum = minimum if minimum is not None else 0.01 # default set to 0.01
		self.gainadapter = numpy.vectorize(lambda x, y, z: z + self.tau if x * y > 0.0 else z * (1.0 - self.tau))
		self.gainclipper = numpy.vectorize(lambda x: self.minimum if x < self.minimum else self.maximum if self.maximum < x else x)
		self.gainweights = numpy.ones(self.modifier.layer.weights.shape, dtype = float)
		self.gainbiases = numpy.ones(self.modifier.layer.biases.shape, dtype = float)
		self.olddeltaweights = numpy.copy(self.modifier.layer.deltaweights)
		self.olddeltabiases = numpy.copy(self.modifier.layer.deltabiases)


	def updateweights(self, deltaweights, deltabiases):
		self.gainweights = self.gainclipper(self.gainadapter(self.olddeltaweights, self.modifier.layer.deltaweights, self.gainweights))
		self.gainbiases = self.gainclipper(self.gainadapter(self.olddeltabiases, self.modifier.layer.deltabiases, self.gainbiases))
		self.olddeltaweights = numpy.copy(self.modifier.layer.deltaweights)
		self.oldbiases = numpy.copy(self.modifier.layer.deltabiases)
		deltaweights = numpy.multiply(self.gainweights, deltaweights)
		deltabiases = numpy.multiply(self.gainbiases, deltabiases)
		return deltaweights, deltabiases

	def trainingsetup(self):
		self.__init__(self.modifier, self.tau, self.maximum, self.minimum)

	def testingsetup(self):
		self.gainadapter = None
		self.gainclipper = None
		self.gainweights = None
		self.gainbiases = None
		self.olddeltaweights = None
		self.olddeltabiases = None

class RootMeanSquarePropagation:

	epsilon = 0.0001

	def __init__(self, modifier, meu = None):
		self.modifier = modifier
		self.meu = meu if meu is not None else 0.9 # default set to 0.9
		self.meansquareweights = numpy.zeros(self.modifier.layer.weights.shape, dtype = float)
		self.meansquarebiases = numpy.zeros(self.modifier.layer.biases.shape, dtype = float)

	def updateweights(self, deltaweights, deltabiases):
		self.meansquareweights = numpy.add(numpy.multiply(self.meu, self.meansquareweights), numpy.multiply(1.0 - self.meu, numpy.square(self.modifier.layer.deltaweights)))
		self.meansquarebiases = numpy.add(numpy.multiply(self.meu, self.meansquarebiases), numpy.multiply(1.0 - self.meu, numpy.square(self.modifier.layer.deltabiases)))
		deltaweights = numpy.divide(deltaweights, numpy.sqrt(numpy.add(RootMeanSquarePropagation.epsilon, self.meansquareweights)))
		deltabiases = numpy.divide(deltabiases, numpy.sqrt(numpy.add(RootMeanSquarePropagation.epsilon, self.meansquarebiases)))
		return deltaweights, deltabiases

	def trainingsetup(self):
		self.__init__(self.modifier, self.meu)

	def testingsetup(self):
		self.meansquareweights = None
		self.meansquarebiases = None
