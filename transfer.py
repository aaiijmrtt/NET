import numpy, math

class Sigmoid:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	function = None
	derivative = None
	hadamard = None

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs
		self.function = numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))
		self.derivative = numpy.vectorize(lambda x: x * (1.0 - x))
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previousoutput))

class HyperbolicTangent:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	function = None
	derivative = None
	hadamard = None

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs
		self.function = numpy.vectorize(lambda x: math.tanh(x))
		self.derivative = numpy.vectorize(lambda x: 1.0 - x * x)
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previousoutput))

class HardHyperbolicTangent:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	function = None
	derivative = None
	hadamard = None

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs
		self.function = numpy.vectorize(lambda x: 1.0 if x > 1.0 else -1.0 if x < -1.0 else x)
		self.derivative = numpy.vectorize(lambda x: 1.0 if -1.0 < x < 1.0 else 0.0)
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previousinput))

class RectifiedLinearUnit:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	function = None
	derivative = None
	hadamard = None

	def __init__(self, inputs):
		self.inputs = inputs
		self.outputs = self.inputs
		self.function = numpy.vectorize(lambda x: 0.0 if x < 0.0 else x)
		self.derivative = numpy.vectorize(lambda x: 0.0 if x < 0.0 else 1.0)
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previousinput))

class ParametricRectifiedLinearUnit:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	parameter = None

	function = None
	derivative = None
	hadamard = None

	def __init__(self, inputs, parameter = None):
		self.inputs = inputs
		self.outputs = self.inputs
		self.parameter = parameter if parameter is not None else 0.01 # default set at 0.01
		self.function = numpy.vectorize(lambda x: self.parameter * x if x < 0.0 else x)
		self.derivative = numpy.vectorize(lambda x: self.parameter if x < 0.0 else 1.0)
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previousinput))

class HardShrink:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	parameter = None

	function = None
	derivative = None
	hadamard = None

	def __init__(self, inputs, parameter = None):
		self.inputs = inputs
		self.outputs = self.inputs
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else x)
		self.derivative = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else 1.0)
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previousinput))

class SoftShrink:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	parameter = None

	function = None
	derivative = None
	hadamard = None

	def __init__(self, inputs, parameter = None):
		self.inputs = inputs
		self.outputs = self.inputs
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: x - self.parameter if x > self.parameter else x + self.parameter if x < -self.parameter else 0.0)
		self.derivative = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else 1.0)
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previousinput))

class SoftMax:

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	hadamard = None
	exponential = None

	def __init__(self, inputs, parameter = None):
		self.inputs = inputs
		self.outputs = self.inputs
		self.exponential = numpy.vectorize(lambda x: math.exp(x))
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		inputvector = numpy.subtract(inputvector, numpy.amax(inputvector))
		inputvector = self.exponential(inputvector)
		self.previousoutput = numpy.divide(inputvector, numpy.sum(inputvector))
		return self.previousoutput

	def backpropagate(self, outputvector):
		had = self.hadamard(self.previousoutput, outputvector)
		dot = numpy.dot(numpy.dot(self.previousoutput, self.previousoutput.transpose()), outputvector)
		return numpy.subtract(had, dot)

class SoftPlus:

	inputs = None
	outputs = None

	previousinput = None
	previoushidden = None
	previousoutput = None

	parameter = None

	function = None
	derivative = None
	hadamard = None
	exponential = None

	def __init__(self, inputs, parameter = None):
		self.inputs = inputs
		self.outputs = self.inputs
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: 1.0 / self.parameter * math.log(x))
		self.derivative = numpy.vectorize(lambda x: 1.0 - 1.0 / x)
		self.exponential = numpy.vectorize(lambda x: 1.0 + math.exp(self.parameter * x))
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def cleardeltas(self):
		pass

	def updateweights(self):
		pass

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previoushidden = self.exponential(self.previousinput)
		self.previousoutput = self.function(self.previoushidden)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previoushidden))
