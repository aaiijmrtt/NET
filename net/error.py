import numpy

class Error:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	derivative = None

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = inputvector
		return self.previousoutput

	def backpropagate(self, outputvector):
		self.previousoutput = outputvector
		return self.derivative(self.previousinput, self.previousoutput)
		
class MeanSquared(Error):

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs
		self.derivative = numpy.vectorize(lambda x, y: x - y)
