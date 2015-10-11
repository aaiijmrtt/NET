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
		deltaparameters = dict()
		for parameter in self.layer.deltaparameters:
			deltaparameters[parameter] = numpy.multiply(learningrate, self.layer.deltaparameters[parameter])
		if self.rootmeansquarepropagation is not None:
			deltaparameters = self.rootmeansquarepropagation.updateweights(deltaparameters)
		if self.adaptivegradient is not None:
			deltaparameters = self.adaptivegradient.updateweights(deltaparameters)
		if self.adaptivegain is not None:
			deltaparameters = self.adaptivegain.updateweights(deltaparameters)
		if self.velocity is not None:
			deltaparameters = self.velocity.updateweights(deltaparameters)
		return deltaparameters

	def cleardeltas(self):
		if self.regularization is not None:
			deltaparameters = self.regularization.cleardeltas()
		else:
			deltaparameters = dict()
			for parameter in self.layer.parameters:
				deltaparameters[parameter] = numpy.zeros(self.layer.parameters[parameter].shape, dtype = float)
		return deltaparameters

	def feedforward(self, inputvector):
		if self.dropout is not None:
			inputvector = self.dropout.feedforward(inputvector)
		return inputvector

	def trainingsetup(self):
		for attribute in ['decay', 'dropout', 'velocity', 'adaptivegain', 'adaptivegradient', 'rootmeansquarepropagation']:
			if self.__dict__[attribute] is not None:
				self.__dict__[attribute].trainingsetup()

	def testingsetup(self):
		for attribute in ['decay', 'dropout', 'velocity', 'adaptivegain', 'adaptivegradient', 'rootmeansquarepropagation']:
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
		self.__init__(self.modifier, self.eta)

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
		deltaparameters = dict()
		for parameter in self.modifier.layer.parameters:
			deltaparameters[parameter] = numpy.multiply(self.lamda, self.regularizer(self.modifier.layer.parameters[parameter]))
		return deltaparameters

class Velocity:

	def __init__(self, modifier, gamma = None):
		self.modifier = modifier
		self.velocityparameters = dict()
		for parameter in self.modifier.layer.parameters:
			self.velocityparameters[parameter] = numpy.zeros(self.modifier.layer.parameters[parameter].shape, dtype = float)
		self.gamma = gamma if gamma is not None else 0.5 # default set at 0.5

	def updateweights(self, deltaparameters):
		for parameter in deltaparameters:
			self.velocityparameters[parameter] = numpy.add(numpy.multiply(self.gamma, self.velocityparameters[parameter]), deltaparameters[parameter])
			deltaparameters[parameter] = numpy.copy(self.velocityparameters[parameter])
		return deltaparameters

	def trainingsetup(self):
		self.__init__(self.modifier, self.gamma)

	def testingsetup(self):
		self.velocityparameters = None

class AdaptiveGradient:

	epsilon = 0.0001

	def __init__(self, modifier):
		self.modifier = modifier
		self.sumsquareparameters = dict()
		for parameter in self.modifier.layer.parameters:
			self.sumsquareparameters[parameter] = numpy.zeros(self.modifier.layer.parameters[parameter].shape, dtype = float)

	def updateweights(self, deltaparameters):
		for parameter in deltaparameters:
			self.sumsquareparameters[parameter] = numpy.add(self.sumsquareparameters[parameter], numpy.square(self.modifier.layer.deltaparameters[parameter]))
			deltaparameters[parameter] = numpy.divide(deltaparameters[parameter], numpy.sqrt(numpy.add(AdaptiveGradient.epsilon, self.sumsquareparameters[parameter])))
		return deltaparameters

	def trainingsetup(self):
		self.__init__(self.modifier)

	def testingsetup(self):
		self.sumsquareparameters = None

class AdaptiveGain:

	def __init__(self, modifier, tau = None, maximum = None, minimum = None):
		self.modifier = modifier
		self.tau = tau if tau is not None else 0.05 # default set to 0.05
		self.maximum = maximum if maximum is not None else 100.0 # default set to 100.0
		self.minimum = minimum if minimum is not None else 0.01 # default set to 0.01
		self.gainadapter = numpy.vectorize(lambda x, y, z: z + self.tau if x * y > 0.0 else z * (1.0 - self.tau))
		self.gainclipper = numpy.vectorize(lambda x: self.minimum if x < self.minimum else self.maximum if self.maximum < x else x)
		self.gainparameters = dict()
		self.olddeltaparameters = dict()
		for parameter in self.modifier.layer.parameters:
			self.gainparameters[parameter] = numpy.ones(self.modifier.layer.parameters[parameter].shape, dtype = float)
			self.olddeltaparameters[parameter] = numpy.copy(self.modifier.layer.deltaparameters[parameter])

	def updateweights(self, deltaparameters):
		for parameter in deltaparameters:
			self.gainparameters[parameter] = self.gainclipper(self.gainadapter(self.olddeltaparameters[parameter], self.modifier.layer.deltaparameters[parameter], self.gainparameters[parameter]))
			self.olddeltaparameters[parameter] = numpy.copy(self.modifier.layer.deltaparameters[parameter])
			deltaparameters[parameter] = numpy.multiply(self.gainparameters[parameter], deltaparameters[parameter])
		return deltaparameters

	def trainingsetup(self):
		self.__init__(self.modifier, self.tau, self.maximum, self.minimum)

	def testingsetup(self):
		self.gainadapter = None
		self.gainclipper = None
		self.gainparameters = None
		self.olddeltaparameters = None

class RootMeanSquarePropagation:

	epsilon = 0.0001

	def __init__(self, modifier, meu = None):
		self.modifier = modifier
		self.meu = meu if meu is not None else 0.9 # default set to 0.9
		self.meansquareparameters = dict()
		for parameter in self.modifier.layer.parameters:
			self.meansquareparameters[parameter] = numpy.zeros(self.modifier.layer.parameters[parameter].shape, dtype = float)

	def updateweights(self, deltaparameters):
		for parameter in deltaparameters:
			self.meansquareparameters[parameter] = numpy.add(numpy.multiply(self.meu, self.meansquareparameters[parameter]), numpy.multiply(1.0 - self.meu, numpy.square(self.modifier.layer.deltaparameters[parameter])))
			deltaparameters[parameter] = numpy.divide(deltaparameters[parameter], numpy.sqrt(numpy.add(RootMeanSquarePropagation.epsilon, self.meansquareparameters[parameter])))
		return deltaparameters

	def trainingsetup(self):
		self.__init__(self.modifier, self.meu)

	def testingsetup(self):
		self.meansquareparameters = None
