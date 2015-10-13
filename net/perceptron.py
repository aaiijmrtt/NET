'''
	Module containing Perceptron Layers.
	Classes embody Parametric Thresholded Binary Layers,
	used to model linear functions.
'''
import numpy
from . import layer, transfer

class Perceptron(layer.Layer):
	'''
		Perceptron Layer
		Mathematically, f(x)(i) = 1.0 if g(x)(i) > random()
								= 0.0 otherwise
						g(x) = W * x + b
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		layer.Layer.__init__(self, inputs, outputs, alpha)
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, 1))
		self.transfer = transfer.Threshold(self.outputs)
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previousoutput = self.transfer.feedforward(numpy.add(numpy.dot(self.parameters['weights'], self.previousinput), self.parameters['biases']))
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		outputvector = self.transfer.backpropagate(outputvector)
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
		self.deltaparameters['biases'] = numpy.add(self.deltaparameters['biases'], outputvector)
		return numpy.dot(numpy.transpose(self.parameters['weights']), outputvector)
