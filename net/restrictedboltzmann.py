'''
	Module containing Self Organising Feature Maps.
	Classes embody Parametric Non Linear Layers,
	used to learn low dimensional representations of data.
'''
import math, numpy
from . import configure, layer, transfer, error

class RestrictedBoltzmannMachine(layer.Layer):
	'''
		Restricted Boltzmann Machine Layer
		Mathematically, f(x) = W' * g(x) + b2
						g(x)(i) = 1 / (1 + exp(-h(x)(i)))
						h(x) = W * x + b1
	'''
	def __init__(self, inputs, hiddens, alpha = None, nonlinearity = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		layer.Layer.__init__(self, inputs, hiddens, alpha)
		self.hiddens = hiddens
		self.previoushidden = list()
		self.parameters['weightsin'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.inputs), (self.hiddens, self.inputs))
		self.parameters['weightsout'] = numpy.transpose(self.parameters['weightsin'])
		self.parameters['biasesin'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.inputs), (self.hiddens, 1))
		self.parameters['biasesout'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.hiddens), (self.inputs, 1))
		self.transferin = nonlinearity() if nonlinearity is not None else transfer.StochasticThreshold(self.inputs)
		self.transferout = nonlinearity() if nonlinearity is not None else transfer.StochasticThreshold(self.inputs)
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(self.modifier.feedforward(inputvector))
		self.previousoutput.append(self.transferin.feedforward(configure.functions['add'](configure.functions['dot'](self.parameters['weightsin'], self.previousinput[-1]), self.parameters['biasesin'])))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		outputvector = self.transferin.backpropagate(outputvector)
		self.deltaparameters['weightsin'] = configure.functions['add'](self.deltaparameters['weightsin'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.previousinput.pop())))
		self.deltaparameters['biasesin'] = configure.functions['add'](self.deltaparameters['biasesin'], outputvector)
		return configure.functions['dot'](configure.functions['transpose'](self.parameters['weightsin']), outputvector)

	def pretrain(self, trainingset, batch = 1, iterations = 1, criterion = None):
		'''
			Method to pretrain parameters using Contrastive Divergence
			: param trainingset : unsupervised training set
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param criterion : criterion used to quantify reconstruction error
			: returns : elementwise reconstruction error on termination
		'''
		def _feedforward(self, inputvector):
			self.previousinput.append(self.modifier.feedforward(inputvector))
			self.previoushidden.append(self.transferin.feedforward(configure.functions['add'](configure.functions['dot'](self.parameters['weightsin'], self.previousinput[-1]), self.parameters['biasesin'])))
			self.previousoutput.append(self.transferout.feedforward(configure.functions['add'](configure.functions['dot'](self.parameters['weightsout'], self.previoushidden[-1]), self.parameters['biasesout'])))
			return self.previousoutput[-1]

		def _backpropagate(self, outputvector):
			outputvector = self.transferout.backpropagate(outputvector)
			self.deltaparameters['weightsout'] = configure.functions['add'](self.deltaparameters['weightsout'], configure.functions['dot'](self.previousoutput[-1], configure.functions['transpose'](self.previoushidden[-1])))
			self.deltaparameters['biasesout'] = configure.functions['add'](self.deltaparameters['biasesout'], self.previousoutput.pop())
			outputvector = self.transferin.backpropagate(configure.functions['dot'](configure.functions['transpose'](self.parameters['weightsout']), outputvector))
			self.deltaparameters['weightsin'] = configure.functions['subtract'](self.deltaparameters['weightsin'], configure.functions['dot'](self.previoushidden[-1], configure.functions['transpose'](self.previousinput.pop())))
			self.deltaparameters['biasesin'] = configure.functions['add'](self.deltaparameters['biasesin'], self.previoushidden.pop())
			return configure.functions['dot'](configure.functions['transpose'](self.parameters['weightsin']), outputvector)

		if criterion is None:
			criterion = error.MeanSquared(self.inputs)
		self.trainingsetup()
		for i in range(iterations):
			for j in range(len(trainingset)):
				if j % batch == 0:
					self.updateweights()
				_feedforward(self, trainingset[j])
				_backpropagate(self, trainingset[j])
		self.testingsetup()
		errorvector = numpy.zeros((self.inputs, 1), dtype = float)
		for vector in trainingset:
			errorvector = configure.functions['add'](errorvector, criterion.compute(_feedforward(self, vector), vector))
		errorvector = configure.functions['divide'](errorvector, len(trainingset))
		return errorvector
