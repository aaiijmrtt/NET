'''
	Module containing Modifiers.
	Classes embody Gradient Descent Optimizations.
'''
import numpy

class Modifier:
	'''
		Class handling all Standard Algorith Modifications
		Mathematically, w(t + 1) = w(t) - p * (dE(t) / dw(t))
	'''
	def __init__(self, layer):
		'''
			Constructor
			: param layer : layer to which modifiers are to be applied
		'''
		self.layer = layer
		self.alpha = None
		self.decay = None
		self.dropout = None
		self.regularization = None
		self.velocity = None
		self.adaptivegain = None
		self.rootmeansquarepropagation = None
		self.adaptivegradient = None
		self.resilientpropagation = None

	def applylearningrate(self, alpha = None):
		'''
			Method to apply learning gradient descent optimization
			: param alpha : p, as given in its mathematical expression
		'''
		self.alpha = alpha if alpha is not None else 0.05 # default set at 0.05

	def applydecayrate(self, eta = None):
		'''
			Method to apply decay gradient descent optimization
			: param eta : decay rate constant hyperparameter
		'''
		self.decay = Decay(self, eta)

	def applyvelocity(self, gamma = None):
		'''
			Method to apply velocity gradient descent optimization
			: param gamma : velocity rate constant hyperparameter
		'''
		self.velocity = Velocity(self, gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
			: param regularizer : regularization function hyperparameter
		'''
		self.regularization = Regularization(self, lamda, regularizer)

	def applydropout(self, rho = None):
		'''
			Method to apply dropout to prevent overfitting
			: param rho : dropout rate constant hyperparameter
		'''
		self.dropout = Dropout(self, rho)

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply adaptive gain gradient descent optimization
			: param tau : adaptive gain rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		self.adaptivegain = AdaptiveGain(self, tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		'''
			Method to apply root mean square propagation gradient descent optimization
			: param meu : root mean square propagation rate constant hyperparameter
		'''
		self.rootmeansquarepropagation = RootMeanSquarePropagation(self, meu)

	def applyadaptivegradient(self):
		'''
			Method to apply adaptive gradient gradient descent optimization
		'''
		self.adaptivegradient = AdaptiveGradient(self)

	def applyresilientpropagation(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply resilient propagation gradient descent optimization
			: param tau : resilient propagation rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		self.resilientpropagation = ResilientPropagation(self, tau, maximum, minimum)

	def updateweights(self):
		'''
			Method to update weights based on accumulated parameter changes
		'''
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
		if self.resilientpropagation is not None:
			deltaparameters = self.resilientpropagation.updateweights(deltaparameters)
		return deltaparameters

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		if self.regularization is not None:
			deltaparameters = self.regularization.cleardeltas()
		else:
			deltaparameters = dict()
			for parameter in self.layer.parameters:
				deltaparameters[parameter] = numpy.zeros(self.layer.parameters[parameter].shape, dtype = float)
		return deltaparameters

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if self.dropout is not None:
			inputvector = self.dropout.feedforward(inputvector)
		return inputvector

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		for attribute in ['decay', 'dropout', 'velocity', 'adaptivegain', 'adaptivegradient', 'rootmeansquarepropagation', 'resilientpropagation']:
			if self.__dict__[attribute] is not None:
				self.__dict__[attribute].trainingsetup()

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		for attribute in ['decay', 'dropout', 'velocity', 'adaptivegain', 'adaptivegradient', 'rootmeansquarepropagation', 'resilientpropagation']:
			if self.__dict__[attribute] is not None:
				self.__dict__[attribute].testingsetup()

class Decay:
	'''
		Decay Modifier Class
		Mathematically, w(t + 1) = w(t) - p1 / (1 + t * p2) * (dE(t) / dw(t))
	'''
	def __init__(self, modifier, eta = None):
		'''
			Constructor
			: param modifier : modifier to which decay is to be applied
			: param eta : p2, as given in its mathematical expression
		'''
		self.modifier = modifier
		self.eta = eta if eta is not None else 0.05 # default set at 0.05
		self.updates = 0

	def updateweights(self, learningrate):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		learningrate /= (1.0 + self.updates * self.eta)
		self.updates += 1
		return learningrate

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.modifier, self.eta)

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.updates = None

class Dropout:
	'''
		Dropout Modifier Class
		Mathematically, during training: x(i) = x(i) if random() > p
												0 otherwise
						during testing: x(i) = p * x(i)
	'''
	def __init__(self, modifier, rho = None):
		'''
			Constructor
			: param modifier : modifier to which dropout is to be applied
			: param rho : p, as given in its mathematical expression
		'''
		self.modifier = modifier
		self.rho = rho if rho is not None else 0.75 # default set at 0.75
		self.training = True

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if self.training:
			return numpy.multiply(numpy.random.binomial(1, self.rho, inputvector.shape), inputvector)
		else:
			return numpy.multiply(self.rho, inputvector)

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.modifier, self.rho)

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.training = False

class Regularization:
	'''
		Regularization Modifier Class
		Mathematically, E = E + p * f(w)
	'''
	L1regularizer = numpy.vectorize(lambda x: 1.0 if x > 0.0 else -1.0 if x < 0.0 else 0.0)
	L2regularizer = numpy.vectorize(lambda x: x)

	def __init__(self, modifier, lamda = None, regularizer = None):
		'''
			Constructor
			: param modifier : modifier to which regularization is to be applied
			: param lamda : p, as given in its mathematical expression
			: param regularizer : f, as given in its mathematical expression
		'''
		self.modifier = modifier
		self.lamda = lamda if lamda is not None else 0.005 # default set at 0.005
		self.regularizer = regularizer if regularizer is not None else Regularization.L2regularizer # default set to L2 regularization

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		deltaparameters = dict()
		for parameter in self.modifier.layer.parameters:
			deltaparameters[parameter] = numpy.multiply(self.lamda, self.regularizer(self.modifier.layer.parameters[parameter]))
		return deltaparameters

class Velocity:
	'''
		Velocity Modifier Class
		Mathematically, v(t + 1) = p1 * v(t) + p2 * (dE(t) / dw(t))
						w(t + 1) = w(t) - v(t)
	'''
	def __init__(self, modifier, gamma = None):
		'''
			Constructor
			: param modifier : modifier to which velocity is to be applied
			: param gamma : p1, as given in its mathematical expression
		'''
		self.modifier = modifier
		self.velocityparameters = dict()
		for parameter in self.modifier.layer.parameters:
			self.velocityparameters[parameter] = numpy.zeros(self.modifier.layer.parameters[parameter].shape, dtype = float)
		self.gamma = gamma if gamma is not None else 0.5 # default set at 0.5

	def updateweights(self, deltaparameters):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.velocityparameters[parameter] = numpy.add(numpy.multiply(self.gamma, self.velocityparameters[parameter]), deltaparameters[parameter])
			deltaparameters[parameter] = numpy.copy(self.velocityparameters[parameter])
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.modifier, self.gamma)

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.velocityparameters = None

class AdaptiveGradient:
	'''
		Adaptive Gradient Modifier Class
		Mathematically, sw(t + 1) = sw(t) + (dE(t) / dw(t)) ^ 2
						w(t + 1) = w(t) - p / (sw(t + 1) + e) ^ 0.5 * (dE(t) / dw(t))
	'''
	epsilon = 0.0001

	def __init__(self, modifier):
		'''
			Constructor
			: param modifier : modifier to which adaptive gradient is to be applied
		'''
		self.modifier = modifier
		self.sumsquareparameters = dict()
		for parameter in self.modifier.layer.parameters:
			self.sumsquareparameters[parameter] = numpy.zeros(self.modifier.layer.parameters[parameter].shape, dtype = float)

	def updateweights(self, deltaparameters):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.sumsquareparameters[parameter] = numpy.add(self.sumsquareparameters[parameter], numpy.square(self.modifier.layer.deltaparameters[parameter]))
			deltaparameters[parameter] = numpy.divide(deltaparameters[parameter], numpy.sqrt(numpy.add(AdaptiveGradient.epsilon, self.sumsquareparameters[parameter])))
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.modifier)

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.sumsquareparameters = None

class AdaptiveGain:
	'''
		Adaptive Gain Modifier Class
		Mathematically, g(t + 1) = g(t) + p1 if (dE(t) / dw(t)) * (dE(t - 1) / dw(t - 1)) > 0
									(1 - p1) * g(t) otherwise
						w(t + 1) = w(t) - p2 * g(t) * dE(t) / dw(t)
	'''
	def __init__(self, modifier, tau = None, maximum = None, minimum = None):
		'''
			Constructor
			: param modifier : modifier to which adaptive gain is to be applied
			: param tau : p1, as given in its mathematical expression
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
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
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.gainparameters[parameter] = self.gainclipper(self.gainadapter(self.olddeltaparameters[parameter], self.modifier.layer.deltaparameters[parameter], self.gainparameters[parameter]))
			self.olddeltaparameters[parameter] = numpy.copy(self.modifier.layer.deltaparameters[parameter])
			deltaparameters[parameter] = numpy.multiply(self.gainparameters[parameter], deltaparameters[parameter])
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.modifier, self.tau, self.maximum, self.minimum)

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.gainadapter = None
		self.gainclipper = None
		self.gainparameters = None
		self.olddeltaparameters = None

class ResilientPropagation:
	'''
		Resilient Propagation Modifier Class
		Mathematically, g(t + 1) = g(t) + p1 if (dE(t) / dw(t)) * (dE(t - 1) / dw(t - 1)) > 0
									(1 - p1) * g(t) otherwise
						w(t + 1) = w(t) - p2 * g(t) * sign(dE(t) / dw(t))
	'''
	def __init__(self, modifier, tau = None, maximum = None, minimum = None):
		'''
			Constructor
			: param modifier : modifier to which aresilient propagation is to be applied
			: param tau : p1, as given in its mathematical expression
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
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
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.gainparameters[parameter] = self.gainclipper(self.gainadapter(self.olddeltaparameters[parameter], self.modifier.layer.deltaparameters[parameter], self.gainparameters[parameter]))
			self.olddeltaparameters[parameter] = numpy.copy(self.modifier.layer.deltaparameters[parameter])
			deltaparameters[parameter] = numpy.multiply(self.gainparameters[parameter], numpy.sign(self.modifier.layer.deltaparameters[parameter]))
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.modifier, self.tau, self.maximum, self.minimum)

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.gainadapter = None
		self.gainclipper = None
		self.gainparameters = None
		self.olddeltaparameters = None

class RootMeanSquarePropagation:
	'''
		Root Mean Square Modifier Class
		Mathematically, msw(t + 1) = p1 * msw(t) + (1 - p1) * (dE(t) / dw(t)) ^ 2
						w(t + 1) = w(t) - p2 / (msw(t + 1) + e) ^ 0.5 * (dE(t) / dw(t))
	'''
	epsilon = 0.0001

	def __init__(self, modifier, meu = None):
		'''
			Constructor
			: param modifier : modifier to which root mean square propagation is to be applied
			: param meu : p1, as given in its mathematical expression
		'''
		self.modifier = modifier
		self.meu = meu if meu is not None else 0.9 # default set to 0.9
		self.meansquareparameters = dict()
		for parameter in self.modifier.layer.parameters:
			self.meansquareparameters[parameter] = numpy.zeros(self.modifier.layer.parameters[parameter].shape, dtype = float)

	def updateweights(self, deltaparameters):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.meansquareparameters[parameter] = numpy.add(numpy.multiply(self.meu, self.meansquareparameters[parameter]), numpy.multiply(1.0 - self.meu, numpy.square(self.modifier.layer.deltaparameters[parameter])))
			deltaparameters[parameter] = numpy.divide(deltaparameters[parameter], numpy.sqrt(numpy.add(RootMeanSquarePropagation.epsilon, self.meansquareparameters[parameter])))
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.modifier, self.meu)

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.meansquareparameters = None
