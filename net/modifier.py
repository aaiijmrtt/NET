'''
	Module containing Modifiers.
	Classes embody Gradient Descent Optimizations.
'''
import numpy
from . import base, configure

class Modifier(base.Net):
	'''
		Class handling all Standard Algorith Modifications
		Mathematically, w(t + 1) = w(t) - p * (dE(t) / dw(t))
	'''
	def __init__(self, layer):
		'''
			Constructor
			: param layer : layer to which modifiers are to be applied
		'''
		self.backpointer = layer
		self.backpointer.metaparameters['alpha'] = None
		if not hasattr(self, 'units'):
			self.units = dict()
		if not hasattr(self, 'parameters'):
			self.parameters = dict()
		if not hasattr(self, 'metaparameters'):
			self.metaparameters = dict()
		self.__finit__()

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()

	def applylearningrate(self, alpha = None):
		'''
			Method to apply learning gradient descent optimization
			: param alpha : p, as given in its mathematical expression
		'''
		self.backpointer.metaparameters['alpha'] = alpha if alpha is not None else 0.05 # default set at 0.05

	def applydecayrate(self, eta = None):
		'''
			Method to apply decay gradient descent optimization
			: param eta : decay rate constant hyperparameter
		'''
		self.units['decay'] = Decay(self, eta)

	def applyvelocity(self, gamma = None):
		'''
			Method to apply velocity gradient descent optimization
			: param gamma : velocity rate constant hyperparameter
		'''
		self.units['velocity'] = Velocity(self, gamma)

	def applyl1regularization(self, lamda = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
		'''
		self.units['regularization'] = L1Regularization(self, lamda)

	def applyl2regularization(self, lamda = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
		'''
		self.units['regularization'] = L2Regularization(self, lamda)

	def applydropout(self, rho = None):
		'''
			Method to apply dropout to prevent overfitting
			: param rho : dropout rate constant hyperparameter
		'''
		self.units['dropout'] = Dropout(self, rho)

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply adaptive gain gradient descent optimization
			: param tau : adaptive gain rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		self.units['adaptivegain'] = AdaptiveGain(self, tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		'''
			Method to apply root mean square propagation gradient descent optimization
			: param meu : root mean square propagation rate constant hyperparameter
		'''
		self.units['rootmeansquarepropagation'] = RootMeanSquarePropagation(self, meu)

	def applyadaptivegradient(self):
		'''
			Method to apply adaptive gradient gradient descent optimization
		'''
		self.units['adaptivegradient'] = AdaptiveGradient(self)

	def applyresilientpropagation(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply resilient propagation gradient descent optimization
			: param tau : resilient propagation rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		self.units['resilientpropagation'] = ResilientPropagation(self, tau, maximum, minimum)

	def updateweights(self):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		learningrate = self.backpointer.metaparameters['alpha']
		if 'decay' in self.units:
			learningrate = self.units['decay'].updateweights(learningrate)
		deltaparameters = dict()
		for parameter in self.backpointer.deltaparameters:
			deltaparameters[parameter] = configure.functions['multiply'](learningrate, self.backpointer.deltaparameters[parameter])
		if 'rootmeansquarepropagation' in self.units:
			deltaparameters = self.units['rootmeansquarepropagation'].updateweights(deltaparameters)
		if 'adaptivegradient' in self.units:
			deltaparameters = self.units['adaptivegradient'].updateweights(deltaparameters)
		if 'adaptivegain' in self.units:
			deltaparameters = self.units['adaptivegain'].updateweights(deltaparameters)
		if 'velocity' in self.units:
			deltaparameters = self.units['velocity'].updateweights(deltaparameters)
		if 'resilientpropagation' in self.units:
			deltaparameters = self.units['resilientpropagation'].updateweights(deltaparameters)
		return deltaparameters

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		if 'regularization' in self.units:
			deltaparameters = self.units['regularization'].cleardeltas()
		else:
			deltaparameters = dict()
			for parameter in self.backpointer.parameters:
				deltaparameters[parameter] = numpy.zeros(self.backpointer.parameters[parameter].shape, dtype = float)
		return deltaparameters

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if 'dropout' in self.units:
			inputvector = self.units['dropout'].feedforward(inputvector)
		return inputvector

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		for unit in self.units:
			self.units[unit].trainingsetup()

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		for unit in self.units:
			self.units[unit].trainingsetup()

class Decay(base.Net):
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
		self.backpointer = modifier
		self.backpointer.metaparameters['eta'] = eta if eta is not None else 0.05 # default set at 0.05
		self.backpointer.metaparameters['updates'] = 0

	def updateweights(self, learningrate):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		learningrate /= (1.0 + self.backpointer.metaparameters['updates'] * self.backpointer.metaparameters['eta'])
		self.backpointer.metaparameters['updates'] += 1
		return learningrate

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.backpointer, self.backpointer.metaparameters['eta'])

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.backpointer.metaparameters['updates'] = None

class Dropout(base.Net):
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
		self.backpointer = modifier
		self.backpointer.metaparameters['rho'] = rho if rho is not None else 0.75 # default set at 0.75
		self.backpointer.metaparameters['training'] = True

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if self.backpointer.metaparameters['training']:
			return configure.functions['multiply'](numpy.random.binomial(1, self.backpointer.metaparameters['rho'], inputvector.shape), inputvector)
		else:
			return configure.functions['multiply'](self.backpointer.metaparameters['rho'], inputvector)

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.backpointer, self.backpointer.metaparameters['rho'])

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.backpointer.metaparameters['training'] = False

class L1Regularization(base.Net):
	'''
		Regularization Modifier Class
		Mathematically, E = E + sum(|w|)
	'''
	def __init__(self, modifier, lamda = None):
		'''
			Constructor
			: param modifier : modifier to which regularization is to be applied
			: param lamda : p, as given in its mathematical expression
		'''
		self.backpointer = modifier
		self.backpointer.metaparameters['lamda'] = lamda if lamda is not None else 0.005 # default set at 0.005
		self.__finit__()

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self.backpointer, 'functions'):
			self.backpointer.functions = dict()
		self.backpointer.functions['regularizer'] = configure.functions['vectorize'](lambda x: 1.0 if x > 0.0 else -1.0 if x < 0.0 else 0.0)

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		deltaparameters = dict()
		for parameter in self.backpointer.backpointer.parameters:
			deltaparameters[parameter] = configure.functions['multiply'](self.backpointer.metaparameters['lamda'], self.backpointer.functions['regularizer'](self.backpointer.backpointer.parameters[parameter]))
		return deltaparameters

class L2Regularization(base.Net):
	'''
		Regularization Modifier Class
		Mathematically, E = E + sum(0.5 * w ^ 2)
	'''

	def __init__(self, modifier, lamda = None):
		'''
			Constructor
			: param modifier : modifier to which regularization is to be applied
			: param lamda : p, as given in its mathematical expression
		'''
		self.backpointer = modifier
		self.backpointer.metaparameters['lamda'] = lamda if lamda is not None else 0.005 # default set at 0.005
		self.__finit__()

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self.backpointer, 'functions'):
			self.backpointer.functions = dict()
		self.backpointer.functions['regularizer'] = configure.functions['vectorize'](lambda x: x)

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		deltaparameters = dict()
		for parameter in self.backpointer.backpointer.parameters:
			deltaparameters[parameter] = configure.functions['multiply'](self.backpointer.metaparameters['lamda'], self.backpointer.functions['regularizer'](self.backpointer.backpointer.parameters[parameter]))
		return deltaparameters

class Velocity(base.Net):
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
		self.backpointer = modifier
		if 'velocity' not in self.backpointer.parameters:
			self.backpointer.parameters['velocity'] = dict()
		for parameter in self.backpointer.backpointer.parameters:
			self.backpointer.parameters['velocity'][parameter] = numpy.zeros(self.backpointer.backpointer.parameters[parameter].shape, dtype = float)
		self.backpointer.metaparameters['gamma'] = gamma if gamma is not None else 0.5 # default set at 0.5

	def updateweights(self, deltaparameters):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.backpointer.parameters['velocity'][parameter] = configure.functions['add'](configure.functions['multiply'](self.backpointer.metaparameters['gamma'], self.backpointer.parameters['velocity'][parameter]), deltaparameters[parameter])
			deltaparameters[parameter] = numpy.copy(self.backpointer.parameters['velocity'][parameter])
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.backpointer, self.backpointer.metaparameters['gamma'])

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.backpointer.parameters['velocity'] = None

class AdaptiveGradient(base.Net):
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
		self.backpointer = modifier
		if 'sumsquare' not in self.backpointer.parameters:
			self.backpointer.parameters['sumsquare'] = dict()
		for parameter in self.backpointer.backpointer.parameters:
			self.backpointer.parameters['sumsquare'][parameter] = numpy.zeros(self.backpointer.backpointer.parameters[parameter].shape, dtype = float)

	def updateweights(self, deltaparameters):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.backpointer.parameters['sumsquare'][parameter] = configure.functions['add'](self.backpointer.parameters['sumsquare'][parameter], configure.functions['square'](self.backpointer.backpointer.deltaparameters[parameter]))
			deltaparameters[parameter] = configure.functions['divide'](deltaparameters[parameter], configure.functions['sqrt'](configure.functions['add'](AdaptiveGradient.epsilon, self.backpointer.parameters['sumsquare'][parameter])))
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.backpointer)

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.backpointer.parameters['sumsquare'] = None

class AdaptiveGain(base.Net):
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
		self.backpointer = modifier
		self.backpointer.metaparameters['tau'] = tau if tau is not None else 0.05 # default set to 0.05
		self.backpointer.metaparameters['maximum'] = maximum if maximum is not None else 100.0 # default set to 100.0
		self.backpointer.metaparameters['minimum'] = minimum if minimum is not None else 0.01 # default set to 0.01
		self.__finit__()
		if 'gain' not in self.backpointer.parameters:
			self.backpointer.parameters['gain'] = dict()
		if 'olddelta' not in self.backpointer.parameters:
			self.backpointer.parameters['olddelta'] = dict()
		for parameter in self.backpointer.backpointer.parameters:
			self.backpointer.parameters['gain'][parameter] = numpy.ones(self.backpointer.backpointer.parameters[parameter].shape, dtype = float)
			self.backpointer.parameters['olddelta'][parameter] = numpy.copy(self.backpointer.backpointer.deltaparameters[parameter])

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self.backpointer, 'functions'):
			self.backpointer.functions = dict()
		self.backpointer.functions['gainadapter'] = configure.functions['vectorize'](lambda x, y, z: z + self.backpointer.metaparameters['tau'] if x * y > 0.0 else z * (1.0 - self.backpointer.metaparameters['tau']))
		self.backpointer.functions['gainclipper'] = configure.functions['vectorize'](lambda x: self.backpointer.metaparameters['minimum'] if x < self.backpointer.metaparameters['minimum'] else self.backpointer.metaparameters['maximum'] if self.backpointer.metaparameters['maximum'] < x else x)

	def updateweights(self, deltaparameters):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.backpointer.parameters['gain'][parameter] = self.backpointer.functions['gainclipper'](self.backpointer.functions['gainadapter'](self.backpointer.parameters['olddelta'][parameter], self.backpointer.backpointer.deltaparameters[parameter], self.backpointer.parameters['gain'][parameter]))
			self.backpointer.parameters['olddelta'][parameter] = numpy.copy(self.backpointer.backpointer.deltaparameters[parameter])
			deltaparameters[parameter] = configure.functions['multiply'](self.backpointer.parameters['gain'][parameter], deltaparameters[parameter])
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.backpointer, self.backpointer.metaparameters['tau'], self.backpointer.metaparameters['maximum'], self.backpointer.metaparameters['minimum'])

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.backpointer.functions['gainadapter'] = None
		self.backpointer.functions['gainclipper'] = None
		self.backpointer.parameters['gain'] = None
		self.backpointer.parameters['olddelta'] = None

class ResilientPropagation(base.Net):
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
		self.backpointer = modifier
		self.backpointer.metaparameters['tau'] = tau if tau is not None else 0.05 # default set to 0.05
		self.backpointer.metaparameters['maximum'] = maximum if maximum is not None else 100.0 # default set to 100.0
		self.backpointer.metaparameters['minimum'] = minimum if minimum is not None else 0.01 # default set to 0.01
		self.__finit__()
		if 'gain' not in self.backpointer.parameters:
			self.backpointer.parameters['gain'] = dict()
		if 'olddelta' not in self.backpointer.parameters:
			self.backpointer.parameters['olddelta'] = dict()
		for parameter in self.backpointer.backpointer.parameters:
			self.backpointer.parameters['gain'][parameter] = numpy.ones(self.backpointer.backpointer.parameters[parameter].shape, dtype = float)
			self.backpointer.parameters['olddelta'][parameter] = numpy.copy(self.backpointer.backpointer.deltaparameters[parameter])

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self.backpointer, 'functions'):
			self.backpointer.functions = dict()
		self.backpointer.functions['gainadapter'] = configure.functions['vectorize'](lambda x, y, z: z + self.backpointer.metaparameters['tau'] if x * y > 0.0 else z * (1.0 - self.backpointer.metaparameters['tau']))
		self.backpointer.functions['gainclipper'] = configure.functions['vectorize'](lambda x: self.backpointer.metaparameters['minimum'] if x < self.backpointer.metaparameters['minimum'] else self.backpointer.metaparameters['maximum'] if self.backpointer.metaparameters['maximum'] < x else x)

	def updateweights(self, deltaparameters):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.backpointer.parameters['gain'][parameter] = self.backpointer.functions['gainclipper'](self.backpointer.functions['gainadapter'](self.backpointer.parameters['olddelta'][parameter], self.backpointer.backpointer.deltaparameters[parameter], self.backpointer.parameters['gain'][parameter]))
			self.backpointer.parameters['olddelta'][parameter] = numpy.copy(self.backpointer.backpointer.deltaparameters[parameter])
			deltaparameters[parameter] = configure.functions['multiply'](self.backpointer.parameters['gain'][parameter], configure.functions['sign'](self.backpointer.backpointer.deltaparameters[parameter]))
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.backpointer, self.backpointer.metaparameters['tau'], self.backpointer.metaparameters['maximum'], self.backpointer.metaparameters['minimum'])

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.backpointer.functions['gainadapter'] = None
		self.backpointer.functions['gainclipper'] = None
		self.backpointer.parameters['gain'] = None
		self.backpointer.parameters['olddelta'] = None

class RootMeanSquarePropagation(base.Net):
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
		self.backpointer = modifier
		self.backpointer.metaparameters['meu'] = meu if meu is not None else 0.9 # default set to 0.9
		if 'meansquare' not in self.backpointer.parameters:
			self.backpointer.parameters['meansquare'] = dict()
		for parameter in self.backpointer.backpointer.parameters:
			self.backpointer.parameters['meansquare'][parameter] = numpy.zeros(self.backpointer.backpointer.parameters[parameter].shape, dtype = float)

	def updateweights(self, deltaparameters):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for parameter in deltaparameters:
			self.backpointer.parameters['meansquare'][parameter] = configure.functions['add'](configure.functions['multiply'](self.backpointer.metaparameters['meu'], self.backpointer.parameters['meansquare'][parameter]), configure.functions['multiply'](1.0 - self.backpointer.metaparameters['meu'], configure.functions['square'](self.backpointer.backpointer.deltaparameters[parameter])))
			deltaparameters[parameter] = configure.functions['divide'](deltaparameters[parameter], configure.functions['sqrt'](configure.functions['add'](RootMeanSquarePropagation.epsilon, self.backpointer.parameters['meansquare'][parameter])))
		return deltaparameters

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.__init__(self.backpointer, self.backpointer.metaparameters['meu'])

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.backpointer.parameters['meansquare'] = None
