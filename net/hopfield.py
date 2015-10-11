import numpy
from . import layer, error

class HopfieldNetwork(layer.Layer):

	def __init__(self, inputs, alpha = None):
		layer.Layer.__init__(self, inputs, inputs, alpha)
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.inputs, self.inputs))
		self.parameters['weights'] = numpy.add(self.parameters['weights'], numpy.transpose(self.parameters['weights']))
		self.cleardeltas()

	def cleardeltas(self):
		self.deltaparameters = self.modifier.cleardeltas()
		for i in range(self.inputs):
			self.parameters['weights'][i][i] = 0.0

	def updateweights(self):
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.transpose(self.deltaparameters['weights']))
		self.deltaparameters = self.modifier.updateweights()
		for parameter in self.parameters:
			self.parameters[parameter] = numpy.subtract(self.parameters[parameter], self.deltaparameters[parameter])
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previousoutput = numpy.dot(self.parameters['weights'], self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
		return numpy.dot(numpy.transpose(self.parameters['weights']), outputvector)

	def pretrain(self, trainingset, batch = 1, iterations = 1, criterion = None): # Hebbian Learning
		self.parameters['weights'] = numpy.zeros((self.inputs, self.inputs), dtype = float)
		if criterion is None:
			criterion = error.MeanSquared(self.inputs)
		self.trainingsetup()
		for i in range(iterations):
			for j in range(len(trainingset)):
				if j % batch == 0:
					self.updateweights()
				self.feedforward(trainingset[j])
				self.backpropagate(-trainingset[j])
		self.testingsetup()
		errorvector = numpy.zeros((self.outputs, 1), dtype = float)
		for vector in trainingset:
			errorvector = numpy.add(errorvector, criterion.compute(self.feedforward(vector), vector))
		errorvector = numpy.divide(errorvector, len(trainingset))
		return errorvector
