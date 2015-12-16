'''
	Module containing Transfer Functions.
	Classes embody Non (Learnable) Parametric Non Linear Layers,
	usually an elementwise mapping.
'''
import math, numpy
from . import base, configure

class Transfer(base.Net):
	'''
		Base Class for Transfer Functions
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
		if not hasattr(self, 'metaparameters'):
			self.metaparameters = dict()
		self.__finit__()

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = None
		self.functions['derivative'] = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['output'].append(self.functions['function'](self.history['input'][-1]))
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].pop()
		return configure.functions['multiply'](outputvector, self.functions['derivative'](self.history['output'].pop()))

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
		self.metaparameters['upper'] = upper if upper is not None else 1.0
		self.metaparameters['lower'] = lower if lower is not None else 0.0

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: self.metaparameters['upper'] if x >= 0.0 else self.metaparameters['lower'])
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 1.0) # invisible during backpropagation

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
		self.metaparameters['upper'] = upper if upper is not None else 1.0
		self.metaparameters['lower'] = lower if lower is not None else 0.0

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: self.metaparameters['upper'] if x >= numpy.random.random() else self.metaparameters['lower'])
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 1.0) # invisible during backpropagation

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

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: 1.0 / (1.0 + math.exp(-x)))
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: x * (1.0 - x))

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

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['tanh']
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 1.0 - x * x)

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

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: 1.0 if x > 1.0 else -1.0 if x < -1.0 else x)
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 1.0 if -1.0 < x < 1.0 else 0.0)

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

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: 0.0 if x < 0.0 else x)
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 0.0 if x < 0.0 else 1.0)

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
		self.metaparameters['parameter'] = parameter if parameter is not None else 0.01 # default set at 0.01

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: self.metaparameters['parameter'] * x if x < 0.0 else x)
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: self.metaparameters['parameter'] if x < 0.0 else 1.0)

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
		self.metaparameters['parameter'] = parameter if parameter is not None else 1.0 # default set at 1.0

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: 0.0 if -self.metaparameters['parameter'] < x < self.metaparameters['parameter'] else x)
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 0.0 if -self.metaparameters['parameter'] < x < self.metaparameters['parameter'] else 1.0)

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
		self.metaparameters['parameter'] = parameter if parameter is not None else 1.0 # default set at 1.0

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: x - self.metaparameters['parameter'] if x > self.metaparameters['parameter'] else x + self.metaparameters['parameter'] if x < -self.metaparameters['parameter'] else 0.0)
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 0.0 if -self.metaparameters['parameter'] < x < self.metaparameters['parameter'] else 1.0)

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
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		inputvector = configure.functions['subtract'](inputvector, configure.functions['amax'](inputvector))
		inputvector = configure.functions['exp'](inputvector)
		self.history['output'].append(configure.functions['divide'](inputvector, configure.functions['sum'](inputvector)))
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		had = configure.functions['multiply'](self.history['output'][-1], outputvector)
		dot = configure.functions['dot'](configure.functions['dot'](self.history['output'][-1], configure.functions['transpose'](self.history['output'][-1])), outputvector)
		self.history['output'].pop()
		self.history['input'].pop()
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
		self.metaparameters['parameter'] = parameter if parameter is not None else 1.0 # default set at 1.0
		self.history['hidden'] = list()

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: 1.0 / self.metaparameters['parameter'] * math.log(x))
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 1.0 - 1.0 / x)
		self.functions['exponential'] = configure.functions['vectorize'](lambda x: 1.0 + math.exp(self.metaparameters['parameter'] * x))

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['hidden'].append(self.functions['exponential'](self.history['input'][-1]))
		self.history['output'].append(self.functions['function'](self.history['hidden'][-1]))
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['output'].pop()
		self.history['input'].pop()
		return configure.functions['multiply'](outputvector, self.functions['derivative'](self.history['hidden'].pop()))

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
		self.metaparameters['scale'] = scale if scale is not None else 1.0
		self.metaparameters['shift'] = shift if shift is not None else 0.0

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: self.metaparameters['scale'] * x + self.metaparameters['shift'])
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: self.metaparameters['scale'])

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

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['vectorize'](lambda x: x / (1.0 + math.fabs(x)))
		self.functions['derivative'] = configure.functions['vectorize'](lambda x: 1.0 / (1.0 + math.fabs(x)) ** 2)

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['output'].pop()
		return configure.functions['multiply'](outputvector, self.functions['derivative'](self.history['input'].pop()))
