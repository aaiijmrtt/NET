'''
	Module containing Containers.
	Classes embody Containers,
	used to define combinations of Layers.
'''
import numpy

class Container:
	'''
		Base class for all Containers
	'''
	listoflayers = ['Linear', 'Normalizer', 'Convolutional', 'Perceptron', 'AutoEncoder', 'HopfieldNetwork', 'GaussianRB', 'MultiQuadraticRB', 'InverseMultiQuadraticRB', 'ThinPlateSplineRB', 'CubicRB', 'LinearRB', 'RestrictedBoltzmann', 'ManhattanSO', 'EuclideanSquaredSO']
	listofcontainers = ['Series', 'Parallel', 'Recurrent']

	def __init__(self):
		'''
			Constructor
		'''
		self.layers = list()
		self.inputs = None
		self.outputs = None
		self.previousinput = None
		self.previousoutput = None

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
			if layer.__class__.__name__ in Container.listofcontainers:
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

	def applyregularization(self, lamda = None, regularizer = None):
		'''
			Method to apply regularization to prevent overfitting
			: param lamda : regularization rate constant hyperparameter
			: param regularizer : regularization function hyperparameter
		'''
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyregularization(lamda, regularizer)

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
		self.outputs = self.layers[-1].outputs
		if self.inputs is None:
			self.inputs = self.layers[-1].inputs

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previousoutput = self.previousinput
		for layer in self.layers:
			self.previousoutput = layer.feedforward(self.previousoutput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
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
		self.inputdimensions = [0]
		self.outputdimensions = [0]
		self.inputs = 0
		self.outputs = 0

	def addlayer(self, layer):
		'''
			Method to add layer to container
			: param layer : layer to be added to container
		'''
		self.layers.append(layer)
		self.inputs += self.layers[-1].inputs
		self.outputs += self.layers[-1].outputs
		self.inputdimensions.append(self.inputdimensions[-1] + self.layers[-1].inputs)
		self.outputdimensions.append(self.outputdimensions[-1] + self.layers[-1].outputs)

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		layeroutputs = list()
		for layer, layerinput in zip(self.layers, numpy.split(self.previousinput, self.inputdimensions[1: -1])):
			layeroutputs.append(layer.feedforward(layerinput))
		self.previousoutput = numpy.concatenate(layeroutputs)
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		layerdeltas = list()
		for layer, layeroutput in zip(self.layers, numpy.split(outputvector, self.outputdimensions[1: -1])):
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
		self.hiddens = hiddens
		self.inputs = self.layers[0].inputs - self.hiddens
		self.outputs = self.layers[0].outputs - self.hiddens
		self.previoushiddens = None
		self.previousdeltas = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		outputvector = self.layers[0].feedforward(numpy.concatenate([self.previoushiddens, self.previousinput]))
		self.previoushiddens, self.previousoutput = numpy.split(outputvector, [self.hiddens])
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		deltavector = self.layers[0].backpropagate(numpy.concatenate([numpy.zeros((self.hiddens + self.outputs, 1), dtype = float)]))
		deltahidden, deltainput = numpy.split(deltavector, [self.hiddens])
		return deltainput

	def timingsetup(self):
		'''
			Method to prepare layer for time recurrence
		'''
		self.previoushiddens = numpy.zeros((self.hiddens, 1), dtype = float)
		Container.timingsetup(self)
