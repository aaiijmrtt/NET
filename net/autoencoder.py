'''
	Module containing Auto Encoder Layers.
	Classes embody Parametric Non Linear Compression Layers,
	used to learn low dimensional representations of data.
'''
import numpy
from . import layer, transfer, error

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
		layer.Layer.__init__(self, inputs, inputs, alpha)
		self.hiddens = hiddens
		self.previoushidden = None
		self.parameters['weightsin'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.hiddens, self.inputs))
		self.parameters['weightsout'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.hiddens), (self.inputs, self.hiddens))
		self.parameters['biasesin'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.hiddens, 1))
		self.parameters['biasesout'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.hiddens), (self.inputs, 1))
		self.transfer = nonlinearity() if nonlinearity is not None else transfer.Sigmoid(self.inputs)
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previoushidden = self.transfer.feedforward(numpy.add(numpy.dot(self.parameters['weightsin'], self.previousinput), self.parameters['biasesin']))
		self.previousoutput = self.previoushidden
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		outputvector = self.transfer.backpropagate(outputvector)
		self.deltaparameters['weightsin'] = numpy.add(self.deltaparameters['weightsin'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
		self.deltaparameters['biasesin'] = numpy.add(self.deltaparameters['biasesin'], outputvector)
		return numpy.dot(numpy.transpose(self.parameters['weightsin']), outputvector)

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
			self.previousinput = self.modifier.feedforward(inputvector)
			self.previoushidden = self.transfer.feedforward(numpy.add(numpy.dot(self.parameters['weightsin'], self.previousinput), self.parameters['biasesin']))
			self.previousoutput = numpy.add(numpy.dot(self.parameters['weightsout'], self.previoushidden), self.parameters['biasesout'])
			return self.previousoutput

		def _backpropagate(self, outputvector):
			self.deltaparameters['weightsout'] = numpy.add(self.deltaparameters['weightsout'], numpy.dot(outputvector, numpy.transpose(self.previoushidden)))
			self.deltaparameters['biasesout'] = numpy.add(self.deltaparameters['biasesout'], outputvector)
			outputvector = self.transfer.backpropagate(numpy.dot(numpy.transpose(self.parameters['weightsout']), outputvector))
			self.deltaparameters['weightsin'] = numpy.add(self.deltaparameters['weightsin'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
			self.deltaparameters['biasesin'] = numpy.add(self.deltaparameters['biasesin'], outputvector)
			return numpy.dot(numpy.transpose(self.parameters['weightsin']), outputvector)

		if criterion is None:
			criterion = error.MeanSquared(self.inputs)
		self.trainingsetup()
		for i in range(iterations):
			for j in range(len(trainingset)):
				if j % batch == 0:
					self.updateweights()
				criterion.feedforward(_feedforward(self, trainingset[j]))
				_backpropagate(self, criterion.backpropagate(trainingset[j]))
		self.testingsetup()
		errorvector = numpy.zeros((self.outputs, 1), dtype = float)
		for vector in trainingset:
			errorvector = numpy.add(errorvector, criterion.compute(_feedforward(self, vector), vector))
		errorvector = numpy.divide(errorvector, len(trainingset))
		return errorvector
