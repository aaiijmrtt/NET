'''
	Module containing Layers.
	Classes embody Parametric Layers,
	used to construct a neural network.
'''
import numpy
from . import modifier

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
		self.previousinput = None
		self.previousoutput = None

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
			self.parameters[parameter] = numpy.subtract(self.parameters[parameter], self.deltaparameters[parameter])
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
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, 1))
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previousoutput = numpy.add(numpy.dot(self.parameters['weights'], self.previousinput), self.parameters['biases'])
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
		self.deltaparameters['biases'] = numpy.add(self.deltaparameters['biases'], outputvector)
		return numpy.dot(numpy.transpose(self.parameters['weights']), outputvector)

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

	def accumulate(self, inputvector):
		'''
			Method to accumulate a vector in a batch of samples
			: param inputvector : vector in input feature space
		'''
		self.linearsum = numpy.add(self.linearsum, inputvector)
		self.quadraticsum = numpy.add(self.quadraticsum, numpy.square(inputvector))
		self.batch += 1

	def feedforward(self, inputvector): # ignores dropout
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previousnormalized = numpy.divide(numpy.subtract(self.previousinput, self.mean), numpy.sqrt(numpy.add(Normalizer.epsilon, self.variance)))
		self.previousoutput = numpy.add(numpy.multiply(self.parameters['weights'], self.previousnormalized), self.parameters['biases'])
		return self.previousoutput

	def backpropagate(self, outputvector): # ignores dropout
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.multiply(outputvector, self.previousnormalized))
		self.deltaparameters['biases'] = numpy.add(self.deltaparameters['biases'], outputvector)
		return numpy.multiply(numpy.divide(self.parameters['weights'], self.batch), numpy.divide(numpy.subtract(self.batch - 1, numpy.square(self.previousnormalized)), numpy.sqrt(numpy.add(Normalizer.epsilon, self.variance))))

	def normalize(self):
		'''
			Method to calculate mean, variance of accumulated vectors in a batch of samples
		'''
		self.mean = numpy.divide(self.linearsum, self.batch)
		self.variance = numpy.subtract(numpy.divide(self.quadraticsum, self.batch), numpy.square(self.mean))

	def accumulatingsetup(self):
		'''
			Method to prepare layer for accumulating vectors in a batch of samples
		'''
		self.batch = 0
		self.linearsum = numpy.zeros((self.inputs, 1), dtype = float)
		self.quadraticsum = numpy.zeros((self.inputs, 1), dtype = float)
