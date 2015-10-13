'''
	Module containing Transfer Functions.
	Classes embody Non (Learnable) Parametric Non Linear Layers,
	usually an elementwise mapping.
'''
import numpy

class Transfer:
	'''
		Base Class for Transfer Functions
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		self.inputs = inputs
		self.outputs = self.inputs
		self.previousinput = None
		self.previousoutput = None
		self.function = None
		self.derivative = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previousoutput = self.function(self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		return numpy.multiply(outputvector, self.derivative(self.previousoutput))

class Threshold(Transfer):
	'''
		Threshold Transfer Function
		Mathematically, f(x)(i) = 1.0 if x(i) > 0.0
								= 0.0 otherwise
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 1.0 if x > 0.0 else 0.0)
		self.derivative = numpy.vectorize(lambda x: 1.0) # invisible during backpropagation

class StochasticThreshold(Transfer):
	'''
		Stochastic Threshold Transfer Function
		Mathematically, f(x)(i) = 1.0 if x(i) > random()
								= 0.0 otherwise
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 1.0 if x > numpy.random.random() else 0.0)
		self.derivative = numpy.vectorize(lambda x: 1.0) # invisible during backpropagation

class Sigmoid(Transfer):
	'''
		Sigmoid Transfer Function
		Mathematically, f(x)(i) = 1 / (1 + exp(-x(i)))
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 1.0 / (1.0 + numpy.exp(-x)))
		self.derivative = numpy.vectorize(lambda x: x * (1.0 - x))

class HyperbolicTangent(Transfer):
	'''
		Hyperbolic Tangent Transfer Function
		Mathematically, f(x)(i) = (exp(x(i)) - exp(-x(i))) / (exp(x(i)) + exp(-x(i)))
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		Transfer.__init__(self, inputs)
		self.function = numpy.tanh
		self.derivative = numpy.vectorize(lambda x: 1.0 - x * x)

class HardHyperbolicTangent(Transfer):
	'''
		Hard Hyperbolic Tangent Transfer Function
		Mathematically,	f(x)(i) = x(i) if |x(i)| < 1
								= |x(i)| / x(i) otherwise
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 1.0 if x > 1.0 else -1.0 if x < -1.0 else x)
		self.derivative = numpy.vectorize(lambda x: 1.0 if -1.0 < x < 1.0 else 0.0)

class RectifiedLinearUnit(Transfer):
	'''
		Rectified Linear Unit Transfer Function
		Mathematically, f(x)(i) = x(i) if x(i) > 0
								= 0 otherwise
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: 0.0 if x < 0.0 else x)
		self.derivative = numpy.vectorize(lambda x: 0.0 if x < 0.0 else 1.0)

class ParametricRectifiedLinearUnit(Transfer):
	'''
		Parametric Rectified Linear Unit Transfer Function
		Mathematically, f(x)(i) = x(i) if x(i) > 0
								= p * x(i) otherwise
	'''
	def __init__(self, inputs, parameter = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param parameter : p, as given in its mathematical expression
		'''
		Transfer.__init__(self, inputs)
		self.parameter = parameter if parameter is not None else 0.01 # default set at 0.01
		self.function = numpy.vectorize(lambda x: self.parameter * x if x < 0.0 else x)
		self.derivative = numpy.vectorize(lambda x: self.parameter if x < 0.0 else 1.0)

class HardShrink(Transfer):
	'''
		Hard Shrink Transfer Function
		Mathematically, f(x)(i) = x(i) if |x(i)| > p
								= 0 otherwise
	'''
	def __init__(self, inputs, parameter = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param parameter : p, as given in its mathematical expression
		'''
		Transfer.__init__(self, inputs)
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else x)
		self.derivative = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else 1.0)

class SoftShrink(Transfer):
	'''
		Soft Shrink Transfer Function
		Mathematically, f(x)(i) = x(i) - |x(i)| / x(i) * p if |x(i)| > p
								= 0 otherwise
	'''
	def __init__(self, inputs, parameter = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param parameter : p, as given in its mathematical expression
		'''
		Transfer.__init__(self, inputs)
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: x - self.parameter if x > self.parameter else x + self.parameter if x < -self.parameter else 0.0)
		self.derivative = numpy.vectorize(lambda x: 0.0 if -self.parameter < x < self.parameter else 1.0)

class SoftMax(Transfer):
	'''
		Soft Max Transfer Function
		Mathematically, f(x)(i) = exp(x(i)) / sum_over_j(exp(x(j)))
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		Transfer.__init__(self, inputs)

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		inputvector = numpy.subtract(inputvector, numpy.amax(inputvector))
		inputvector = numpy.exp(inputvector)
		self.previousoutput = numpy.divide(inputvector, numpy.sum(inputvector))
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		had = numpy.multiply(self.previousoutput, outputvector)
		dot = numpy.dot(numpy.dot(self.previousoutput, numpy.transpose(self.previousoutput)), outputvector)
		return numpy.subtract(had, dot)

class SoftPlus(Transfer):
	'''
		Soft Plus Transfer Function
		Mathematically, f(x)(i) = 1 / p * log(1 + exp(p * x(i)))
	'''
	def __init__(self, inputs, parameter = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param parameter : p, as given in its mathematical expression
		'''
		Transfer.__init__(self, inputs)
		self.parameter = parameter if parameter is not None else 1.0 # default set at 1.0
		self.function = numpy.vectorize(lambda x: 1.0 / self.parameter * numpy.log(x))
		self.derivative = numpy.vectorize(lambda x: 1.0 - 1.0 / x)
		self.exponential = numpy.vectorize(lambda x: 1.0 + numpy.exp(self.parameter * x))

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previoushidden = self.exponential(self.previousinput)
		self.previousoutput = self.function(self.previoushidden)
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		return numpy.multiply(outputvector, self.derivative(self.previoushidden))

class ShiftScale(Transfer):
	'''
		Shift Scale Transfer Function
		Mathematically, f(x)(i) = p1 * x(i) + p2
	'''
	def __init__(self, inputs, scale = None, shift = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param scale : p1, as given in its mathematical expression
			: param shift : p2, as given in its mathematical expression
		'''
		Transfer.__init__(self, inputs)
		self.scale = scale if scale is not None else 1.0
		self.shift = shift if shift is not None else 0.0
		self.function = numpy.vectorize(lambda x: self.scale * x + self.shift)
		self.derivative = numpy.vectorize(lambda x: self.scale)

class SoftSign(Transfer):
	'''
		Shift Scale Transfer Function
		Mathematically, f(x)(i) = 1 / (1 + |x(i)|)
	'''
	def __init__(self, inputs):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
		'''
		Transfer.__init__(self, inputs)
		self.function = numpy.vectorize(lambda x: x / (1.0 + numpy.fabs(x)))
		self.derivative = numpy.vectorize(lambda x: 1.0 / (1.0 + numpy.fabs(x)) ** 2)

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		return numpy.multiply(outputvector, self.derivative(self.previousinput))
