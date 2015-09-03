import numpy

class Transfer:

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
		self.hadamard = numpy.vectorize(lambda x, y: x * y)

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previousoutput))

class Sigmoid(Transfer):

	def __init__(self, inputs):
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 1.0 / (1.0 + numpy.exp(-x)))
		self.derivative = numpy.vectorize(lambda x: x * (1.0 - x))

class HyperbolicTangent(Transfer):

	def __init__(self, inputs):
		Transfer.__init__(self, inputs)
		self.function = numpy.tanh
		self.derivative = numpy.vectorize(lambda x: 1.0 - x * x)

class HardHyperbolicTangent(Transfer):

	def __init__(self, inputs):
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 1.0 if x > 1.0 else -1.0 if x < -1.0 else x)
		self.derivative = numpy.vectorize(lambda x: 1.0 if -1.0 < x < 1.0 else 0.0)

class RectifiedLinearUnit(Transfer):

	def __init__(self, inputs):
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 0.0 if x < 0.0 else x)
		self.derivative = numpy.vectorize(lambda x: 0.0 if x < 0.0 else 1.0)

class ParametricRectifiedLinearUnit(Transfer):

	parameter = None

	def __init__(self, inputs, parameter = None):
		Transfer.__init__(self, inputs)
		self.parameter = parameter if parameter is not None else 0.01 # default set at 0.01
		self.function = numpy.vectorize(lambda x: self.parameter * x if x < 0.0 else x)
		self.derivative = numpy.vectorize(lambda x: self.parameter if x < 0.0 else 1.0)

class HardShrink(Transfer):

	parameter = None

	def __init__(self, inputs, parameter = None):
		Transfer.__init__(self, inputs)
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else x)
		self.derivative = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else 1.0)

class SoftShrink(Transfer):

	parameter = None

	def __init__(self, inputs, parameter = None):
		Transfer.__init__(self, inputs)
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: x - self.parameter if x > self.parameter else x + self.parameter if x < -self.parameter else 0.0)
		self.derivative = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else 1.0)

class SoftMax(Transfer):

	exponential = None

	def __init__(self, inputs, parameter = None):
		Transfer.__init__(self, inputs)
		self.exponential = numpy.exp

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

class SoftPlus(Transfer):

	previoushidden = None
	parameter = None
	exponential = None

	def __init__(self, inputs, parameter = None):
		Transfer.__init__(self, inputs)
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: 1.0 / self.parameter * numpy.log(x))
		self.derivative = numpy.vectorize(lambda x: 1.0 - 1.0 / x)
		self.exponential = numpy.vectorize(lambda x: 1.0 + numpy.exp(self.parameter * x))

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previoushidden = self.exponential(self.previousinput)
		self.previousoutput = self.function(self.previoushidden)
		return self.previousoutput

	def backpropagate(self, outputvector):
		return self.hadamard(outputvector, self.derivative(self.previoushidden))

class ShiftScale(Transfer):

	scale = None
	shift = None

	def __init__(self, inputs, scale = None, shift = None):
		Transfer.__init__(self, inputs)
		self.scale = scale if scale is not None else 1.0
		self.shift = shift if shift is not None else 0.0
		self.function = numpy.vectorize(lambda x: self.scale * x + self.shift)
		self.derivative = numpy.vectorize(lambda x: self.scale)
