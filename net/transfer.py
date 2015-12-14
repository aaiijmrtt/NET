'''
	Module containing Transfer Functions.
	Classes embody Non (Learnable) Parametric Non Linear Layers,
	usually an elementwise mapping.
'''
import math, numpy
from . import configure

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
		self.previousinput = list()
		self.previousoutput = list()
		self.function = None
		self.derivative = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previousoutput.append(self.function(self.previousinput[-1]))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousinput.pop()
		return configure.functions['multiply'](outputvector, self.derivative(self.previousoutput.pop()))

class Threshold(Transfer):
	'''
		Threshold Transfer Function
		Mathematically, f(x)(i) = p1 if x(i) >= 0.0
								= p2 otherwise
	'''
	def __init__(self, inputs, upper = None, lower = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param upper : p1, as given in its mathematical expression
			: param lower : p2, as given in its mathematical expression
		'''
		Transfer.__init__(self, inputs)
		self.upper = upper if upper is not None else 1.0
		self.lower = lower if lower is not None else 0.0
		self.function = configure.functions['vectorize'](lambda x: self.upper if x >= 0.0 else self.lower)
		self.derivative = configure.functions['vectorize'](lambda x: 1.0) # invisible during backpropagation

class StochasticThreshold(Transfer):
	'''
		Stochastic Threshold Transfer Function
		Mathematically, f(x)(i) = p1 if x(i) >= random()
								= p2 otherwise
	'''
	def __init__(self, inputs, upper = None, lower = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param upper : p1, as given in its mathematical expression
			: param lower : p2, as given in its mathematical expression
		'''
		Transfer.__init__(self, inputs)
		self.upper = upper if upper is not None else 1.0
		self.lower = lower if lower is not None else 0.0
		self.function = configure.functions['vectorize'](lambda x: self.upper if x >= numpy.random.random() else self.lower)
		self.derivative = configure.functions['vectorize'](lambda x: 1.0) # invisible during backpropagation

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
		self.function = configure.functions['vectorize'](lambda x: 1.0 / (1.0 + math.exp(-x)))
		self.derivative = configure.functions['vectorize'](lambda x: x * (1.0 - x))

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
		self.function = configure.functions['tanh']
		self.derivative = configure.functions['vectorize'](lambda x: 1.0 - x * x)

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
		self.function = configure.functions['vectorize'](lambda x: 1.0 if x > 1.0 else -1.0 if x < -1.0 else x)
		self.derivative = configure.functions['vectorize'](lambda x: 1.0 if -1.0 < x < 1.0 else 0.0)

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
		self.function = configure.functions['vectorize'](lambda x: 0.0 if x < 0.0 else x)
		self.derivative = configure.functions['vectorize'](lambda x: 0.0 if x < 0.0 else 1.0)

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
		self.function = configure.functions['vectorize'](lambda x: self.parameter * x if x < 0.0 else x)
		self.derivative = configure.functions['vectorize'](lambda x: self.parameter if x < 0.0 else 1.0)

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
		self.function = configure.functions['vectorize'](lambda x: 0.0 if -self.parameter < x < self.parameter else x)
		self.derivative = configure.functions['vectorize'](lambda x: 0.0 if -self.parameter < x < self.parameter else 1.0)

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
		self.function = configure.functions['vectorize'](lambda x: x - self.parameter if x > self.parameter else x + self.parameter if x < -self.parameter else 0.0)
		self.derivative = configure.functions['vectorize'](lambda x: 0.0 if -self.parameter < x < self.parameter else 1.0)

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
		self.previousinput.append(inputvector)
		inputvector = configure.functions['subtract'](inputvector, configure.functions['amax'](inputvector))
		inputvector = configure.functions['exp'](inputvector)
		self.previousoutput.append(configure.functions['divide'](inputvector, configure.functions['sum'](inputvector)))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		had = configure.functions['multiply'](self.previousoutput[-1], outputvector)
		dot = configure.functions['dot'](configure.functions['dot'](self.previousoutput[-1], configure.functions['transpose'](self.previousoutput[-1])), outputvector)
		self.previousoutput.pop()
		self.previousinput.pop()
		return configure.functions['subtract'](had, dot)

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
		self.function = configure.functions['vectorize'](lambda x: 1.0 / self.parameter * math.log(x))
		self.derivative = configure.functions['vectorize'](lambda x: 1.0 - 1.0 / x)
		self.exponential = configure.functions['vectorize'](lambda x: 1.0 + math.exp(self.parameter * x))
		self.previoushidden = list()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previoushidden.append(self.exponential(self.previousinput[-1]))
		self.previousoutput.append(self.function(self.previoushidden[-1]))
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		self.previousinput.pop()
		return configure.functions['multiply'](outputvector, self.derivative(self.previoushidden.pop()))

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
		self.function = configure.functions['vectorize'](lambda x: self.scale * x + self.shift)
		self.derivative = configure.functions['vectorize'](lambda x: self.scale)

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
		self.function = configure.functions['vectorize'](lambda x: x / (1.0 + math.fabs(x)))
		self.derivative = configure.functions['vectorize'](lambda x: 1.0 / (1.0 + math.fabs(x)) ** 2)

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		return configure.functions['multiply'](outputvector, self.derivative(self.previousinput.pop()))
