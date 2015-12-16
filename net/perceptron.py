'''
	Module containing Perceptron Layers.
	Classes embody Parametric Thresholded Binary Layers,
	used to model linear functions.
'''
import math, numpy
from . import configure, layer, transfer

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
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], self.dimensions['inputs']))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], 1))
		if not hasattr(self, 'units'):
			self.units = dict()
		self.units['transfer'] = transfer.Threshold(self.dimensions['outputs'])
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
		self.history['output'].append(self.units['transfer'].feedforward(configure.functions['add'](configure.functions['dot'](self.parameters['weights'], self.history['input'][-1]), self.parameters['biases'])))
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
		self.deltaparameters['weights'] = configure.functions['add'](self.deltaparameters['weights'], configure.functions['dot'](outputvector, configure.functions['transpose'](self.history['input'].pop())))
		self.deltaparameters['biases'] = configure.functions['add'](self.deltaparameters['biases'], outputvector)
		return configure.functions['dot'](configure.functions['transpose'](self.parameters['weights']), outputvector)
