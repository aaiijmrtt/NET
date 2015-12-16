'''
	Module containing Connector Layers.
	Classes embody Non Parametric Layers,
	used to connect combinations of Layers.
'''
import numpy
from . import base

class Split(base.Net):
	'''
		Split Connector Function
		Mathematically, f(x) = [x, x ... x]
	'''
	def __init__(self, inputs, parameter):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param parameter : determines dimension of output feature space,
				as multiplicity of input feature space
		'''
		if not hasattr(self, 'dimensions'):
			self.dimensions = dict()
		self.dimensions['inputs'] = inputs
		self.dimensions['parameter'] = parameter
		self.dimensions['outputs'] = self.dimensions['inputs'] * self.dimensions['parameter']
		if not hasattr(self, 'history'):
			self.history = dict()
		self.history['input'] = list()
		self.history['output'] = list()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['output'].append(numpy.concatenate([self.history['input'][-1]] * self.dimensions['parameter']))
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.history['output'].pop()
		self.history['input'].pop()
		deltas = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
		for i in range(self.dimensions['outputs']):
			deltas[i % self.dimensions['inputs']][0] += outputvector[i][0]
		return deltas

class MergeSum(base.Net):
	'''
		Merge (Sum) Connector Function
		Mathematically, f([x1, x2 .. xn])(i) = sum_over_j(xj(i))
	'''
	def __init__(self, outputs, parameter):
		'''
			Constructor
			: param outputs : dimension of output feature space
			: param parameter : determines dimension of input feature space,
				as multiplicity of output feature space
		'''
		if not hasattr(self, 'dimensions'):
			self.dimensions = dict()
		self.dimensions['outputs'] = outputs
		self.dimensions['parameter'] = parameter
		self.dimensions['inputs'] = self.dimensions['outputs'] * self.dimensions['parameter']
		if not hasattr(self, 'history'):
			self.history = dict()
		self.history['input'] = list()
		self.history['output'] = list()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['output'].append(numpy.zeros((self.dimensions['outputs'], 1), dtype = float))
		for i in range(self.dimensions['inputs']):
			self.history['output'][-1][i % self.dimensions['outputs']][0] += self.history['input'][-1][i][0]
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.history['output'].pop()
		self.history['input'].pop()
		return numpy.concatenate([outputvector] * self.dimensions['parameter'])

class MergeProduct(base.Net):
	'''
		Merge (Product) Connector Function
		Mathematically, f([x1, x2 .. xn])(i) = product_over_j(xj(i))
	'''
	def __init__(self, outputs, parameter):
		'''
			Constructor
			: param outputs : dimension of output feature space
			: param parameter : determines dimension of input feature space,
				as multiplicity of output feature space
		'''
		if not hasattr(self, 'dimensions'):
			self.dimensions = dict()
		self.dimensions['outputs'] = outputs
		self.dimensions['parameter'] = parameter
		self.dimensions['inputs'] = self.dimensions['outputs'] * self.dimensions['parameter']
		if not hasattr(self, 'history'):
			self.history = dict()
		self.history['input'] = list()
		self.history['output'] = list()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['output'].append(numpy.ones((self.dimensions['outputs'], 1), dtype = float))
		for i in range(self.dimensions['inputs']):
			self.history['output'][-1][i % self.dimensions['outputs']][0] *= self.history['input'][-1][i][0]
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		deltas = numpy.concatenate([outputvector] * self.dimensions['parameter'])
		for i in range(self.dimensions['inputs']):
			deltas[i][0] *= self.history['output'][-1][i % self.dimensions['outputs']][0] / self.history['input'][-1][i][0]
		self.history['output'].pop()
		self.history['input'].pop()
		return deltas

class Step(base.Net):
	'''
		Step Connector Function
		Mathematically, f(x) = x
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		if not hasattr(self, 'dimensions'):
			self.dimensions = dict()
		self.dimensions['inputs'] = inputs
		self.dimensions['outputs'] = self.dimensions['inputs']
		if not hasattr(self, 'history'):
			self.history = dict()
		self.history['input'] = list()
		self.history['output'] = list()

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
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.history['input'].pop()
		self.history['output'].pop()
		return outputvector

class Constant(base.Net):
	'''
		Identity Connector Function
		Mathematically, f(x) = p
	'''
	def __init__(self, inputs, outputs, parameter = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param outputs : dimension of output feature space
			: param parameter : p, as given in its mathematical equation
		'''
		if not hasattr(self, 'dimensions'):
			self.dimensions = dict()
		self.dimensions['inputs'] = inputs
		self.dimensions['outputs'] = outputs
		if not hasattr(self, 'history'):
			self.history = dict()
		self.history['input'] = list()
		self.history['output'] = list()
		self.parameter = parameter if parameter is not None else 1.0

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['output'].append(numpy.multiply(self.parameter, numpy.ones((self.dimensions['outputs'], 1), dtype = float)))
		return self.history['output'][-1]

	def backpropagate(self, outputvector): # invisible during backpropagation
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].pop()
		self.history['output'].pop()
		return outputvector
