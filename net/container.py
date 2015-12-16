'''
	Module containing Containers.
	Classes embody Containers,
	used to define combinations of Layers.
'''
import numpy
from . import base

class Container(base.Net):
	'''
		Base class for all Containers
	'''
	listoflayers = ['Linear', 'Normalizer', 'Convolutional', 'Perceptron', 'AutoEncoder', 'HopfieldNetwork', 'GaussianRB', 'MultiQuadraticRB', 'InverseMultiQuadraticRB', 'ThinPlateSplineRB', 'CubicRB', 'LinearRB', 'RestrictedBoltzmann', 'ManhattanSO', 'EuclideanSquaredSO']
	listofcontainers = ['Series', 'Parallel', 'Recurrent']

	def __init__(self):
		'''
			Constructor
		'''
		if not hasattr(self, 'dimensions'):
			self.dimensions = dict()
		self.dimensions['inputs'] = None
		self.dimensions['outputs'] = None
		if not hasattr(self, 'history'):
			self.history = dict()
		self.history['input'] = list()
		self.history['output'] = list()
		self.layers = list()

	def accumulate(self, inputvector):
		'''
			Method to accumulate a vector in a batch of samples
			: param inputvector : vector in input feature space
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in ['Normalizer'] + Container.listofcontainers:
				layer.accumulate(inputvector)

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.cleardeltas()

	def updateweights(self):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.updateweights()

	def normalize(self):
		'''
			Method to calculate mean, variance of accumulated vectors in a batch of samples
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in ['Normalizer'] + Container.listofcontainers:
				layer.normalize()

	def accumulatingsetup(self):
		'''
			Method to prepare layer for accumulating vectors in a batch of samples
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in ['Normalizer'] + Container.listofcontainers:
				layer.accumulatingsetup()

	def timingsetup(self):
		'''
			Method to prepare layer for time recurrence
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in ['LongShortTermMemory'] + Container.listofcontainers:
				layer.timingsetup()

	def trainingsetup(self):
		'''
			Method to prepare layer for training
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.trainingsetup()

	def testingsetup(self):
		'''
			Method to prepare layer for testing
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.trainingsetup()

	def applylearningrate(self, alpha = None):
		'''
			Method to apply learning gradient descent optimization
			: param alpha : learning rate constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applylearningrate(alpha)

	def applydecayrate(self, eta = None):
		'''
			Method to apply decay gradient descent optimization
			: param eta : decay rate constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applydecayrate(eta)

	def applyvelocity(self, gamma = None):
		'''
			Method to apply velocity gradient descent optimization
			: param gamma : velocity rate constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyvelocity(gamma)

	def applyl1regularization(self, lamda = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyl1regularization(lamda)

	def applyl2regularization(self, lamda = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyl2regularization(lamda)

	def applydropout(self, rho = None):
		'''
			Method to apply dropout to prevent overfitting
			: param rho : dropout rate constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applydropout(rho)

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply adaptive gain gradient descent optimization
			: param tau : adaptive gain rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyadaptivegain(tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		'''
			Method to apply root mean square propagation gradient descent optimization
			: param meu : root mean square propagation rate constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyrootmeansquarepropagation(meu)

	def applyadaptivegradient(self):
		'''
			Method to apply adaptive gradient gradient descent optimization
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyadaptivegradient()

	def applyresilientpropagation(self, tau = None, maximum = None, minimum = None):
		'''
			Method to apply resilient propagation gradient descent optimization
			: param tau : resilient propagation rate constant hyperparameter
			: param maximum : maximum gain constant hyperparameter
			: param minimum : minimum gain constant hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyresilientpropagation(tau, maximum, minimum)

class Series(Container):
	'''
		Series Container
		Mathematically, f(x) = fn( ... f2(f1(x)))
	'''
	def __init__(self):
		'''
			Constructor
		'''
		Container.__init__(self)

	def addlayer(self, layer):
		'''
			Method to add layer to container
			: param layer : layer to be added to container
		'''
		self.layers.append(layer)
		self.dimensions['outputs'] = self.layers[-1].dimensions['outputs']
		if self.dimensions['inputs'] is None:
			self.dimensions['inputs'] = self.layers[-1].dimensions['inputs']

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['output'].append(self.history['input'][-1])
		for layer in self.layers:
			self.history['output'][-1] = layer.feedforward(self.history['output'][-1])
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
		for layer in reversed(self.layers):
			outputvector = layer.backpropagate(outputvector)
		return outputvector

class Parallel(Container):
	'''
		Parallel Container
		Mathematically, f([x1, x2 ... xn]) = [f1(x1), f2(x2) ... fn(xn)]
	'''
	def __init__(self):
		'''
			Constructor
		'''
		Container.__init__(self)
		self.dimensions['inputsplit'] = [0]
		self.dimensions['outputsplit'] = [0]
		self.dimensions['inputs'] = 0
		self.dimensions['outputs'] = 0

	def addlayer(self, layer):
		'''
			Method to add layer to container
			: param layer : layer to be added to container
		'''
		self.layers.append(layer)
		self.dimensions['inputs'] += self.layers[-1].dimensions['inputs']
		self.dimensions['outputs'] += self.layers[-1].dimensions['outputs']
		self.dimensions['inputsplit'].append(self.dimensions['inputsplit'][-1] + self.layers[-1].dimensions['inputs'])
		self.dimensions['outputsplit'].append(self.dimensions['outputsplit'][-1] + self.layers[-1].dimensions['outputs'])

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		layeroutputs = list()
		for layer, layerinput in zip(self.layers, numpy.split(self.history['input'][-1], self.dimensions['inputsplit'][1: -1])):
			layeroutputs.append(layer.feedforward(layerinput))
		self.history['output'].append(numpy.concatenate(layeroutputs))
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
		layerdeltas = list()
		for layer, layeroutput in zip(self.layers, numpy.split(outputvector, self.dimensions['outputsplit'][1: -1])):
			layerdeltas.append(layer.backpropagate(layeroutput))
		return numpy.concatenate(layerdeltas)

class Recurrent(Container):
	'''
		Recurrent Container
		Mathematically, F([h(t-1), x(t)]) = [h(t), f(x(t))]
	'''
	def __init__(self, hiddens, layer):
		'''
			Constructor
			: param hiddens : dimension of hidden recurrent feature space
			: param layer : layer on which recurrence is applied
		'''
		Container.__init__(self)
		self.layers.append(layer)
		self.dimensions['hiddens'] = hiddens
		self.dimensions['inputs'] = self.layers[0].dimensions['inputs'] - self.dimensions['hiddens']
		self.dimensions['outputs'] = self.layers[0].dimensions['outputs'] - self.dimensions['hiddens']
		self.history['hiddens'] = [numpy.zeros((self.dimensions['hiddens'], 1), dtype = float)]
		self.history['deltas'] = [numpy.zeros((self.dimensions['hiddens'], 1), dtype = float)]

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		outputvector = self.layers[0].feedforward(numpy.concatenate([self.history['hiddens'][-1], self.history['input'][-1]]))
		hiddens, output = numpy.split(outputvector, [self.dimensions['hiddens']])
		self.history['hiddens'].append(hiddens)
		self.history['output'].append(output)
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
		deltavector = self.layers[0].backpropagate(numpy.concatenate([self.history['deltas'][-1], outputvector]))
		deltahidden, deltainput = numpy.split(deltavector, [self.dimensions['hiddens']])
		self.history['deltas'].append(deltahidden)
		return deltainput

	def timingsetup(self):
		'''
			Method to prepare layer for time recurrence
		'''
		self.history['hiddens'] = [numpy.zeros((self.dimensions['hiddens'], 1), dtype = float)]
		self.history['deltas'] = [numpy.zeros((self.dimensions['hiddens'], 1), dtype = float)]
		Container.timingsetup(self)
