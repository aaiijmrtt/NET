import numpy

class Split:

	inputs = None
	outputs = None

	parameter = None

	previousinput = None
	previousoutput = None

	def __init__(self, inputs, parameter):
		self.inputs = inputs
		self.parameter = parameter
		self.outputs = self.inputs * self.parameter

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = numpy.concatenate([self.previousinput] * self.parameter)
		return self.previousoutput

	def backpropagate(self, outputvector):
		deltas = numpy.zeros((self.inputs, 1), dtype = float)
		for i in range(self.outputs):
			deltas[i % self.inputs][0] += outputvector[i][0]
		return deltas

class MergeSum:

	inputs = None
	outputs = None

	parameter = None

	previousinput = None
	previousoutput = None

	def __init__(self, outputs, parameter):
		self.outputs = outputs
		self.parameter = parameter
		self.inputs = self.outputs * self.parameter

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = numpy.zeros((self.outputs, 1), dtype = float)
		for i in range(self.inputs):
			self.previousoutput[i % self.outputs][0] += self.previousinput[i][0]
		return self.previousoutput

	def backpropagate(self, outputvector):
		return numpy.concatenate([outputvector] * self.parameter)

class MergeProduct:

	inputs = None
	outputs = None

	parameter = None

	previousinput = None
	previousoutput = None

	def __init__(self, outputs, parameter):
		self.outputs = outputs
		self.parameter = parameter
		self.inputs = self.outputs * self.parameter

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = numpy.ones((self.outputs, 1), dtype = float)
		for i in range(self.inputs):
			self.previousoutput[i % self.outputs][0] *= self.previousinput[i][0]
		return self.previousoutput

	def backpropagate(self, outputvector):
		deltas = numpy.concatenate([outputvector] * self.parameter)
		for i in range(self.inputs):
			deltas[i][0] *= self.previousoutput[i % self.outputs][0] / self.previousinput[i][0]
		return deltas

class Step:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.previousinput
		return self.previousoutput

	def backpropagate(self, outputvector):
		return outputvector
