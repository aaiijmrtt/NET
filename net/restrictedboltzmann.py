import numpy
from . import layer, transfer, error

class RestrictedBoltzmann(layer.Layer):

	def __init__(self, inputs, hiddens, alpha = None, nonlinearity = None):
		layer.Layer.__init__(self, inputs, inputs, alpha)
		self.hiddens = hiddens
		self.previoushidden = None
		self.parameters['weightsin'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.hiddens, self.inputs))
		self.parameters['weightsout'] = numpy.transpose(self.parameters['weightsin'])
		self.parameters['biasesin'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.hiddens, 1))
		self.parameters['biasesout'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.hiddens), (self.inputs, 1))
		self.transferin = nonlinearity() if nonlinearity is not None else transfer.StochasticThreshold(self.inputs)
		self.transferout = nonlinearity() if nonlinearity is not None else transfer.StochasticThreshold(self.inputs)
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previoushidden = self.transferin.feedforward(numpy.add(numpy.dot(self.parameters['weightsin'], self.previousinput), self.parameters['biasesin']))
		self.previousoutput = self.previoushidden
		return self.previousoutput

	def backpropagate(self, outputvector):
		outputvector = self.transferin.backpropagate(outputvector)
		self.deltaparameters['weightsin'] = numpy.add(self.deltaparameters['weightsin'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
		self.deltaparameters['biasesin'] = numpy.add(self.deltaparameters['biasesin'], outputvector)
		return numpy.dot(numpy.transpose(self.parameters['weightsin']), outputvector)

	def pretrain(self, trainingset, batch = 1, iterations = 1, criterion = None): # Contrastive Divergence
		def _feedforward(self, inputvector):
			self.previousinput = self.modifier.feedforward(inputvector)
			self.previoushidden = self.transferin.feedforward(numpy.add(numpy.dot(self.parameters['weightsin'], self.previousinput), self.parameters['biasesin']))
			self.previousoutput = self.transferout.feedforward(numpy.add(numpy.dot(self.parameters['weightsout'], self.previoushidden), self.parameters['biasesout']))
			return self.previousoutput

		def _backpropagate(self, outputvector):
			outputvector = self.transferout.backpropagate(outputvector)
			self.deltaparameters['weightsout'] = numpy.add(self.deltaparameters['weightsout'], numpy.dot(self.previousoutput, numpy.transpose(self.previoushidden)))
			self.deltaparameters['biasesout'] = numpy.add(self.deltaparameters['biasesout'], self.previousoutput)
			outputvector = self.transferin.backpropagate(numpy.dot(numpy.transpose(self.parameters['weightsout']), outputvector))
			self.deltaparameters['weightsin'] = numpy.subtract(self.deltaparameters['weightsin'], numpy.dot(self.previoushidden, numpy.transpose(self.previousinput)))
			self.deltaparameters['biasesin'] = numpy.add(self.deltaparameters['biasesin'], self.previoushidden)
			return numpy.dot(numpy.transpose(self.parameters['weightsin']), outputvector)

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
		errorvector = numpy.zeros((self.outputs, 1), dtype = float)
		for vector in trainingset:
			errorvector = numpy.add(errorvector, criterion.compute(_feedforward(self, vector), vector))
		errorvector = numpy.divide(errorvector, len(trainingset))
		return errorvector
