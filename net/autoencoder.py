'''
	Module containing Auto Encoder Layers.
	Classes embody Parametric Non Linear Compression Layers,
	used to learn low dimensional representations of data.
'''
import math, numpy
from . import configure, layer, transfer, error

class AutoEncoder(layer.Layer):
	'''
		Auto Encoder Layer
		Mathematically, f(x) = W2 * g(x) + b2
						g(x)(i) = 1 / (1 + exp(-h(x)(i)))
						h(x) = W1 * x + b1
	'''
	def __init__(self, inputs, hiddens, alpha = None, nonlinearity = None):
		'''
			Constructor
			: param inputs : dimension of input (and reconstructed output) feature space
			: param hiddens : dimension of compressed output feature space
			: param alpha : learning rate constant hyperparameter
			: param nonlinearity : transfer function applied after linear transformation of inputs
		'''
		layer.Layer.__init__(self, inputs, hiddens, alpha)
		self.dimensions['hiddens'] = hiddens
		self.parameters['weightsin'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['hiddens'], self.dimensions['inputs']))
		self.parameters['weightsout'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['hiddens']), (self.dimensions['inputs'], self.dimensions['hiddens']))
		self.parameters['biasesin'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['hiddens'], 1))
		self.parameters['biasesout'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['hiddens']), (self.dimensions['inputs'], 1))
		self.units['transfer'] = nonlinearity() if nonlinearity is not None else transfer.Sigmoid(self.dimensions['hiddens'])
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
		self.history['output'].append(self.units['transfer'].feedforward(configure.functions['add'](configure.functions['dot'](self.parameters['weightsin'], self.history['input'][-1]), self.parameters['biasesin'])))
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
		outputvector = self.units['transfer'].backpropagate(outputvector)
		self.deltaparameters['weightsin'] = configure.functions['add'](self.deltaparameters['weightsin'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.history['input'].pop())))
		self.deltaparameters['biasesin'] = configure.functions['add'](self.deltaparameters['biasesin'], outputvector)
		return configure.functions['dot'](configure.functions['transpose'](self.parameters['weightsin']), outputvector)

	def pretrain(self, trainingset, batch = 1, iterations = 1, criterion = None):
		'''
			Method to pretrain parameters using Gradient Descent
			: param trainingset : unsupervised training set
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param criterion : criterion used to quantify reconstruction error
			: returns : elementwise reconstruction error on termination
		'''
		def _feedforward(self, inputvector):
			self.history['input'].append(self.units['modifier'].feedforward(inputvector))
			self.history['hidden'].append(self.units['transfer'].feedforward(configure.functions['add'](configure.functions['dot'](self.parameters['weightsin'], self.history['input'][-1]), self.parameters['biasesin'])))
			self.history['output'].append(configure.functions['add'](configure.functions['dot'](self.parameters['weightsout'], self.history['hidden'][-1]), self.parameters['biasesout']))
			return self.history['output'][-1]

		def _backpropagate(self, outputvector):
			self.history['output'].pop()
			self.deltaparameters['weightsout'] = configure.functions['add'](self.deltaparameters['weightsout'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.history['hidden'].pop())))
			self.deltaparameters['biasesout'] = configure.functions['add'](self.deltaparameters['biasesout'], outputvector)
			outputvector = self.units['transfer'].backpropagate(configure.functions['dot'](configure.functions['transpose'](self.parameters['weightsout']), outputvector))
			self.deltaparameters['weightsin'] = configure.functions['add'](self.deltaparameters['weightsin'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.history['input'].pop())))
			self.deltaparameters['biasesin'] = configure.functions['add'](self.deltaparameters['biasesin'], outputvector)
			return configure.functions['dot'](configure.functions['transpose'](self.parameters['weightsin']), outputvector)

		if criterion is None:
			criterion = error.MeanSquared(self.inputs)
		self.trainingsetup()
		self.hidden = list()
		for i in range(iterations):
			for j in range(len(trainingset)):
				if j % batch == 0:
					self.updateweights()
				criterion.feedforward(_feedforward(self, trainingset[j]))
				_backpropagate(self, criterion.backpropagate(trainingset[j]))
		self.testingsetup()
		errorvector = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
		for vector in trainingset:
			errorvector = configure.functions['add'](errorvector, criterion.compute(_feedforward(self, vector), vector))
		errorvector = configure.functions['divide'](errorvector, len(trainingset))
		return errorvector
