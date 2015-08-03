import numpy

class MeanSquared:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	derivative = None

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs
		self.derivative = numpy.vectorize(lambda x, y: x - y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = inputvector
		return self.previousoutput

	def backpropagate(self, outputvector):
		self.previousoutput = outputvector
		return self.derivative(self.previousinput, self.previousoutput)
