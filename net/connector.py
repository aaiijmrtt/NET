'''
	Module containing Connector Layers.
	Classes embody Non Parametric Layers,
	used to connect combinations of Layers.
'''
import numpy

class Split:
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
		self.inputs = inputs
		self.parameter = parameter
		self.outputs = self.inputs * self.parameter
		self.previousinput = None
		self.previousoutput = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previousoutput = numpy.concatenate([self.previousinput] * self.parameter)
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		deltas = numpy.zeros((self.inputs, 1), dtype = float)
		for i in range(self.outputs):
			deltas[i % self.inputs][0] += outputvector[i][0]
		return deltas

class MergeSum:
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
		self.outputs = outputs
		self.parameter = parameter
		self.inputs = self.outputs * self.parameter
		self.previousinput = None
		self.previousoutput = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previousoutput = numpy.zeros((self.outputs, 1), dtype = float)
		for i in range(self.inputs):
			self.previousoutput[i % self.outputs][0] += self.previousinput[i][0]
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		return numpy.concatenate([outputvector] * self.parameter)

class MergeProduct:
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
		self.outputs = outputs
		self.parameter = parameter
		self.inputs = self.outputs * self.parameter
		self.previousinput = None
		self.previousoutput = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previousoutput = numpy.ones((self.outputs, 1), dtype = float)
		for i in range(self.inputs):
			self.previousoutput[i % self.outputs][0] *= self.previousinput[i][0]
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		deltas = numpy.concatenate([outputvector] * self.parameter)
		for i in range(self.inputs):
			deltas[i][0] *= self.previousoutput[i % self.outputs][0] / self.previousinput[i][0]
		return deltas

class Step:
	'''
		Step Connector Function
		Mathematically, f(x) = x
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param outputs : dimension of input (and output) feature space
		'''
		self.inputs = inputs
		self.outputs = self.inputs
		self.previousinput = None
		self.previousoutput = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previousoutput = self.previousinput
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		return outputvector
