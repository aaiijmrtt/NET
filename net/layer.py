'''
	Module containing Layers.
	Classes embody Parametric Layers,
	used to construct a neural network.
'''
import math, numpy
from . import base, configure, modifier, transfer

class Layer(base.Net):
	'''
		Base class for all Layers
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		if not hasattr(self, 'dimensions'):
			self.dimensions = dict()
		self.dimensions['inputs'] = inputs
		self.dimensions['outputs'] = outputs
		if not hasattr(self, 'history'):
			self.history = dict()
		self.history['input'] = list()
		self.history['output'] = list()
		if not hasattr(self, 'parameters'):
			self.parameters = dict()
		if not hasattr(self, 'deltaparameters'):
			self.deltaparameters = dict()
		if not hasattr(self, 'metaparameters'):
			self.metaparameters = dict()
		if not hasattr(self, 'units'):
			self.units = dict()
		self.units['modifier'] = modifier.Modifier(self)
		self.applylearningrate(alpha)

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		self.deltaparameters = self.units['modifier'].cleardeltas()

	def updateweights(self):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		self.deltaparameters = self.units['modifier'].updateweights()
		for parameter in self.parameters:
			self.parameters[parameter] = configure.functions['subtract'](self.parameters[parameter], self.deltaparameters[parameter])
		self.cleardeltas()

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.cleardeltas()
		self.units['modifier'].trainingsetup()

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.units['modifier'].testingsetup()

	def applylearningrate(self, alpha = None):
		'''
			Method to apply learning gradient descent optimization
			: param alpha : learning rate constant hyperparameter
		'''
		self.units['modifier'].applylearningrate(alpha)

	def applydecayrate(self, eta = None):
		'''
			Method to apply decay gradient descent optimization
			: param eta : decay rate constant hyperparameter
		'''
		self.units['modifier'].applydecayrate(eta)

	def applyvelocity(self, gamma = None):
		'''
			Method to apply velocity gradient descent optimization
			: param gamma : velocity rate constant hyperparameter
		'''
		self.units['modifier'].applyvelocity(gamma)

	def applyl1regularization(self, lamda = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
		'''
		self.units['modifier'].applyl1regularization(lamda)

	def applyl2regularization(self, lamda = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
		'''
		self.units['modifier'].applyl2regularization(lamda)

	def applydropout(self, rho = None):
		'''
			Method to apply dropout to prevent overfitting
			: param rho : dropout rate constant hyperparameter
		'''
		self.units['modifier'].applydropout(rho)

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply adaptive gain gradient descent optimization
			: param tau : adaptive gain rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		self.units['modifier'].applyadaptivegain(tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		'''
			Method to apply root mean square propagation gradient descent optimization
			: param meu : root mean square propagation rate constant hyperparameter
		'''
		self.units['modifier'].applyrootmeansquarepropagation(meu)

	def applyadaptivegradient(self):
		'''
			Method to apply adaptive gradient gradient descent optimization
		'''
		self.units['modifier'].applyadaptivegradient()

	def applyresilientpropagation(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply resilient propagation gradient descent optimization
			: param tau : resilient propagation rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		self.units['modifier'].applyresilientpropagation(tau, maximum, minimum)

class Linear(Layer):
	'''
		Linear Layer
		Mathematically, f(x) = W * x + b
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : parameter learning rate
		'''
		Layer.__init__(self, inputs, outputs, alpha)
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], self.dimensions['inputs']))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], 1))
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(self.units['modifier'].feedforward(inputvector))
		self.history['output'].append(configure.functions['add'](configure.functions['dot'](self.parameters['weights'], self.history['input'][-1]), self.parameters['biases']))
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['output'].pop()
		self.deltaparameters['weights'] = configure.functions['add'](self.deltaparameters['weights'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.history['input'].pop())))
		self.deltaparameters['biases'] = configure.functions['add'](self.deltaparameters['biases'], outputvector)
		return configure.functions['dot'](configure.functions['transpose'](self.parameters['weights']), outputvector)

class OneHotLinear(Layer):
	'''
		One Hot Linear Layer
		Mathematically, f(x) = W * x + b, when x is a one hot binary vector
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : parameter learning rate
		'''
		Layer.__init__(self, inputs, outputs, alpha)
		self.history['index'] = list()
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], self.dimensions['inputs']))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], 1))
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(self.units['modifier'].feedforward(inputvector))
		self.history['index'].append(configure.functions['argmax'](self.history['inputs']))
		self.history['output'].append(configure.functions['add'](self.parameters['weights'][-1][:, self.history['index'][-1]].reshape(self.dimensions['outputs'], 1) , self.parameters['biases']))
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['output'].pop()
		self.history['input'].pop()
		self.deltaparameters['weights'][:, self.history['index'][-1]] = configure.functions['add'](self.deltaparameters['weights'][:, self.history['index'][-1]], outputvector.reshape(1, self.dimensions['output']))
		self.deltaparameters['biases'] = configure.functions['add'](self.deltaparameters['biases'], outputvector)
		self.history['index'].pop()
		return configure.functions['dot'](configure.functions['transpose'](self.parameters['weights']), outputvector) # unoptimizable (?)

class Nonlinear(Layer):
	'''
		NonLinear Layer
		Mathematically, f(x) = sigma(W * x + b)
	'''
	def __init__(self, inputs, outputs, alpha = None, activation = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : parameter learning rate
			: param activation : sigma, as given in its mathematical equation
		'''
		Layer.__init__(self, inputs, outputs, alpha)
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], self.dimensions['inputs']))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], 1))
		self.units['activation'] = activation(self.dimensions['outputs']) if activation is not None else transfer.Sigmoid(self.dimensions['outputs'])
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(self.units['modifier'].feedforward(inputvector))
		self.history['output'].append(self.units['activation'].feedforward(configure.functions['add'](configure.functions['dot'](self.parameters['weights'], self.history['input'][-1]), self.parameters['biases'])))
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['output'].pop()
		outputvector = self.units['activation'].backpropagate(outputvector)
		self.deltaparameters['weights'] = configure.functions['add'](self.deltaparameters['weights'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.history['input'].pop())))
		self.deltaparameters['biases'] = configure.functions['add'](self.deltaparameters['biases'], outputvector)
		return configure.functions['dot'](configure.functions['transpose'](self.parameters['weights']), outputvector)

class Normalizer(Layer):
	'''
		Normalizer Layer
		Mathematically, f(x)(i) = p1 * (x(i) - m(x(i))) / (v(x(i)) + e) ^ 0.5 + p2
	'''
	epsilon = 0.0001

	def __init__(self, inputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param alpha : parameter learning rate
		'''
		Layer.__init__(self, inputs, inputs, alpha)
		self.parameters['weights'] = numpy.ones((self.dimensions['inputs'], 1), dtype = float)
		self.parameters['biases'] = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
		if not hasattr(self, 'accumulator'):
			self.accumulator = dict()
		self.accumulator['mean'] = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
		self.accumulator['variance'] = numpy.ones((self.dimensions['inputs'], 1), dtype = float)
		self.accumulator['batch'] = 1
		self.cleardeltas()
		self.accumulator['linearsum'] = None
		self.accumulator['quadraticsum'] = None
		self.history['normalized'] = list()

	def accumulate(self, inputvector):
		'''
			Method to accumulate a vector in a batch of samples
			: param inputvector : vector in input feature space
		'''
		self.accumulator['linearsum'] = configure.functions['add'](self.accumulator['linearsum'], inputvector)
		self.accumulator['quadraticsum'] = configure.functions['add'](self.accumulator['quadraticsum'], configure.functions['square'](inputvector))
		self.accumulator['batch'] += 1

	def feedforward(self, inputvector): # ignores dropout
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['normalized'].append(configure.functions['divide'](configure.functions['subtract'](self.history['input'][-1], self.accumulator['mean']), configure.functions['sqrt'](configure.functions['add'](Normalizer.epsilon, self.accumulator['variance']))))
		self.history['output'].append(configure.functions['add'](configure.functions['multiply'](self.parameters['weights'], self.history['normalized'][-1]), self.parameters['biases']))
		return self.history['output'][-1]

	def backpropagate(self, outputvector): # ignores dropout
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['output'].pop()
		self.history['input'].pop()
		self.deltaparameters['weights'] = configure.functions['add'](self.deltaparameters['weights'], configure.functions['multiply'](outputvector, self.history['normalized'][-1]))
		self.deltaparameters['biases'] = configure.functions['add'](self.deltaparameters['biases'], outputvector)
		return configure.functions['multiply'](configure.functions['divide'](self.parameters['weights'], self.accumulator['batch']), configure.functions['divide'](configure.functions['subtract'](self.accumulator['batch'] - 1, configure.functions['square'](self.history['normalized'].pop())), configure.functions['sqrt'](configure.functions['add'](Normalizer.epsilon, self.accumulator['variance']))))

	def normalize(self):
		'''
			Method to calculate mean, variance of accumulated vectors in a batch of samples
		'''
		self.accumulator['mean'] = configure.functions['divide'](self.accumulator['linearsum'], self.accumulator['batch'])
		self.accumulator['variance'] = configure.functions['subtract'](configure.functions['divide'](self.accumulator['quadraticsum'], self.accumulator['batch']), configure.functions['square'](self.accumulator['mean']))

	def accumulatingsetup(self):
		'''
			Method to prepare layer for accumulating vectors in a batch of samples
		'''
		self.accumulator['batch'] = 0
		self.accumulator['linearsum'] = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
		self.accumulator['quadraticsum'] = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
