import numpy
from . import layer, transfer

class Perceptron(layer.Layer):

	def __init__(self, inputs, outputs, alpha = None):
		layer.Layer.__init__(self, inputs, outputs, alpha)
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, 1))
		self.transfer = transfer.Threshold(self.outputs)
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previousoutput = self.transfer.feedforward(numpy.add(numpy.dot(self.parameters['weights'], self.previousinput), self.parameters['biases']))
		return self.previousoutput

	def backpropagate(self, outputvector):
		outputvector = self.transfer.backpropagate(outputvector)
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
		self.deltaparameters['biases'] = numpy.add(self.deltaparameters['biases'], outputvector)
		return numpy.dot(numpy.transpose(self.parameters['weights']), outputvector)
