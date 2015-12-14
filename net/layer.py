'''
	Module containing Layers.
	Classes embody Parametric Layers,
	used to construct a neural network.
'''
import math, numpy
from . import configure, modifier, transfer

class Layer:
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
		self.inputs = inputs
		self.outputs = outputs
		self.modifier = modifier.Modifier(self)
		self.applylearningrate(alpha)
		self.parameters = dict()
		self.deltaparameters = dict()
		self.previousinput = list()
		self.previousoutput = list()

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		self.deltaparameters = self.modifier.cleardeltas()

	def updateweights(self):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		self.deltaparameters = self.modifier.updateweights()
		for parameter in self.parameters:
			self.parameters[parameter] = configure.functions['subtract'](self.parameters[parameter], self.deltaparameters[parameter])
		self.cleardeltas()

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		self.cleardeltas()
		self.modifier.trainingsetup()

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		self.modifier.testingsetup()

	def applylearningrate(self, alpha = None):
		'''
			Method to apply learning gradient descent optimization
			: param alpha : learning rate constant hyperparameter
		'''
		self.modifier.applylearningrate(alpha)

	def applydecayrate(self, eta = None):
		'''
			Method to apply decay gradient descent optimization
			: param eta : decay rate constant hyperparameter
		'''
		self.modifier.applydecayrate(eta)

	def applyvelocity(self, gamma = None):
		'''
			Method to apply velocity gradient descent optimization
			: param gamma : velocity rate constant hyperparameter
		'''
		self.modifier.applyvelocity(gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
			: param regularizer : regularization function hyperparameter
		'''
		self.modifier.applyregularization(lamda, regularizer)

	def applydropout(self, rho = None):
		'''
			Method to apply dropout to prevent overfitting
			: param rho : dropout rate constant hyperparameter
		'''
		self.modifier.applydropout(rho)

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply adaptive gain gradient descent optimization
			: param tau : adaptive gain rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		self.modifier.applyadaptivegain(tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		'''
			Method to apply root mean square propagation gradient descent optimization
			: param meu : root mean square propagation rate constant hyperparameter
		'''
		self.modifier.applyrootmeansquarepropagation(meu)

	def applyadaptivegradient(self):
		'''
			Method to apply adaptive gradient gradient descent optimization
		'''
		self.modifier.applyadaptivegradient()

	def applyresilientpropagation(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply resilient propagation gradient descent optimization
			: param tau : resilient propagation rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		self.modifier.applyresilientpropagation(tau, maximum, minimum)

	def applyquickpropagation(self):
		'''
			Method to apply quick propagation gradient descent optimization
		'''
		self.modifier.applyquickpropagation()

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
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.inputs), (self.outputs, self.inputs))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.inputs), (self.outputs, 1))
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(self.modifier.feedforward(inputvector))
		self.previousoutput.append(configure.functions['add'](configure.functions['dot'](self.parameters['weights'], self.previousinput[-1]), self.parameters['biases']))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		self.deltaparameters['weights'] = configure.functions['add'](self.deltaparameters['weights'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.previousinput.pop())))
		self.deltaparameters['biases'] = configure.functions['add'](self.deltaparameters['biases'], outputvector)
		return configure.functions['dot'](configure.functions['transpose'](self.parameters['weights']), outputvector)

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
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.inputs), (self.outputs, self.inputs))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.inputs), (self.outputs, 1))
		self.activation = activation(self.inputs) if activation is not None else transfer.Sigmoid(self.inputs)
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(self.modifier.feedforward(inputvector))
		self.previousoutput.append(self.activation.feedforward(configure.functions['add'](configure.functions['dot'](self.parameters['weights'], self.previousinput[-1]), self.parameters['biases'])))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		outputvector = self.activation.backpropagate(outputvector)
		self.deltaparameters['weights'] = configure.functions['add'](self.deltaparameters['weights'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.previousinput.pop())))
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
		self.parameters['weights'] = numpy.ones((self.inputs, 1), dtype = float)
		self.parameters['biases'] = numpy.zeros((self.inputs, 1), dtype = float)
		self.mean = numpy.zeros((self.inputs, 1), dtype = float)
		self.variance = numpy.ones((self.inputs, 1), dtype = float)
		self.batch = 1
		self.cleardeltas()
		self.linearsum = None
		self.quadraticsum = None
		self.previousnormalized = list()

	def accumulate(self, inputvector):
		'''
			Method to accumulate a vector in a batch of samples
			: param inputvector : vector in input feature space
		'''
		self.linearsum = configure.functions['add'](self.linearsum, inputvector)
		self.quadraticsum = configure.functions['add'](self.quadraticsum, configure.functions['square'](inputvector))
		self.batch += 1

	def feedforward(self, inputvector): # ignores dropout
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previousnormalized.append(configure.functions['divide'](configure.functions['subtract'](self.previousinput[-1], self.mean), configure.functions['sqrt'](configure.functions['add'](Normalizer.epsilon, self.variance))))
		self.previousoutput.append(configure.functions['add'](configure.functions['multiply'](self.parameters['weights'], self.previousnormalized[-1]), self.parameters['biases']))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector): # ignores dropout
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		self.previousinput.pop()
		self.deltaparameters['weights'] = configure.functions['add'](self.deltaparameters['weights'], configure.functions['multiply'](outputvector, self.previousnormalized[-1]))
		self.deltaparameters['biases'] = configure.functions['add'](self.deltaparameters['biases'], outputvector)
		return configure.functions['multiply'](configure.functions['divide'](self.parameters['weights'], self.batch), configure.functions['divide'](configure.functions['subtract'](self.batch - 1, configure.functions['square'](self.previousnormalized.pop())), configure.functions['sqrt'](configure.functions['add'](Normalizer.epsilon, self.variance))))

	def normalize(self):
		'''
			Method to calculate mean, variance of accumulated vectors in a batch of samples
		'''
		self.mean = configure.functions['divide'](self.linearsum, self.batch)
		self.variance = configure.functions['subtract'](configure.functions['divide'](self.quadraticsum, self.batch), configure.functions['square'](self.mean))

	def accumulatingsetup(self):
		'''
			Method to prepare layer for accumulating vectors in a batch of samples
		'''
		self.batch = 0
		self.linearsum = numpy.zeros((self.inputs, 1), dtype = float)
		self.quadraticsum = numpy.zeros((self.inputs, 1), dtype = float)
