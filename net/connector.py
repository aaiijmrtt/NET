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
		self.previousinput = list()
		self.previousoutput = list()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previousoutput.append(numpy.concatenate([self.previousinput[-1]] * self.parameter))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		self.previousinput.pop()
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
		self.previousinput = list()
		self.previousoutput = list()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previousoutput.append(numpy.zeros((self.outputs, 1), dtype = float))
		for i in range(self.inputs):
			self.previousoutput[-1][i % self.outputs][0] += self.previousinput[-1][i][0]
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		self.previousinput.pop()
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
		self.previousinput = list()
		self.previousoutput = list()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previousoutput.append(numpy.ones((self.outputs, 1), dtype = float))
		for i in range(self.inputs):
			self.previousoutput[-1][i % self.outputs][0] *= self.previousinput[-1][i][0]
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		deltas = numpy.concatenate([outputvector] * self.parameter)
		for i in range(self.inputs):
			deltas[i][0] *= self.previousoutput[-1][i % self.outputs][0] / self.previousinput[-1][i][0]
		self.previousoutput.pop()
		self.previousinput.pop()
		return deltas

class Step:
	'''
		Step Connector Function
		Mathematically, f(x) = x
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		self.inputs = inputs
		self.outputs = self.inputs
		self.previousinput = list()
		self.previousoutput = list()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previousoutput.append(self.previousinput[-1])
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousinput.pop()
		self.previousoutput.pop()
		return outputvector

class Constant:
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
		self.inputs = inputs
		self.outputs = outputs
		self.previousinput = list()
		self.previousoutput = list()
		self.parameter = parameter if parameter is not None else 1.0

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previousoutput.append(numpy.multiply(self.parameter, numpy.ones((self.outputs, 1), dtype = float)))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector): # invisible during backpropagation
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousinput.pop()
		self.previousoutput.pop()
		return outputvector
