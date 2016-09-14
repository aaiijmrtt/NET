'''
	Module containing Convolution Layers.
	Classes embody Spatial Convolutions,
	used to exploit spatial coherence.
'''
import numpy
from . import base, configure, layer

class Convolution(base.Net):
	'''
		Base Class for Spatial Convolution Layers
	'''
	def __init__(self, height, width, depth, extent, stride = None, padding = None):
		'''
			Constructor
			: param height : dimension of input feature space along spatial height
			: param width : dimension of input feature space along spatial width
			: param depth : dimension of input feature space along spatial depth
			: param extent : dimension of input feature space convoluted along spatial dimension
			: param stride : dimension of input feature space skipped between convolutions
			: param padding : dimension of padding added to periphery of spatial dimension
		'''
		if not hasattr(self, 'dimensions'):
			self.dimensions = dict()
		self.dimensions['height'] = height
		self.dimensions['width'] = width
		self.dimensions['depth'] = depth
		self.dimensions['extent'] = extent
		self.dimensions['stride'] = stride if stride is not None else 1
		self.dimensions['padding'] = padding if padding is not None else 0
		self.dimensions['rows'] = (self.dimensions['height'] - self.dimensions['extent'] + 2 * self.dimensions['padding']) / self.dimensions['stride'] + 1
		self.dimensions['columns'] = (self.dimensions['width'] - self.dimensions['extent'] + 2 * self.dimensions['padding']) / self.dimensions['stride'] + 1
		if not hasattr(self, 'history'):
			self.history = dict()

	def convolute(self, outrow, outcolumn, invector):
		'''
			Method to perform spatial convolution during feedforward
			: param outrow : index in output feature space along spatial height
			: param outcolumn : index in output feature space along spatial width
			: param invector : vector in input feature space
			: returns : convoluted vector mapped to output feature space
		'''
		vector = numpy.empty((self.dimensions['extent'] * self.dimensions['extent'] * self.dimensions['depth'], 1), dtype = float)
		for depthextent in range(self.dimensions['depth']):
			for rowextent in range(self.dimensions['extent']):
				inrow = - self.dimensions['padding'] + outrow * self.dimensions['stride'] + rowextent
				for columnextent in range(self.dimensions['extent']):
					incolumn = - self.dimensions['padding'] + outcolumn * self.dimensions['stride'] + columnextent
					if inrow in range(self.dimensions['height']) and incolumn in range(self.dimensions['width']):
						vector[(depthextent * self.dimensions['extent'] + rowextent) * self.dimensions['extent'] + columnextent][0] = invector[(depthextent * self.dimensions['height'] + inrow) * self.dimensions['width'] + incolumn][0]
					else:
						vector[(depthextent * self.dimensions['extent'] + rowextent) * self.dimensions['extent'] + columnextent][0] = 0.0
		return vector

	def unconvolute(self, outrow, outcolumn, outvector, backvector):
		'''
			Method to perform inverse spatial convolution during backpropagate
			: param outrow : index in output feature space along spatial height
			: param outcolumn : index in output feature space along spatial width
			: param outvector : vector in output feature space
			: param backvector : backpropagated vector mapped to input feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		for depthextent in range(self.dimensions['depth']):
			for rowextent in range(self.dimensions['extent']):
				inrow = - self.dimensions['padding'] + outrow * self.dimensions['stride'] + rowextent
				for columnextent in range(self.dimensions['extent']):
					incolumn = - self.dimensions['padding'] + outcolumn * self.dimensions['stride'] + columnextent
					if inrow in range(self.dimensions['height']) and incolumn in range(self.dimensions['width']):
						backvector[(depthextent * self.dimensions['height'] + inrow) * self.dimensions['width'] + incolumn][0] += outvector[(depthextent * self.dimensions['extent'] + rowextent) * self.dimensions['extent'] + columnextent][0]
		return backvector

class Convolutional(layer.Layer, Convolution):
	'''
		Convolutional Layer
		Mathematically, f(x) = [W * conv(x) + b]
	'''
	def __init__(self, height, width, depth, extent, alpha = None, stride = None, padding = None):
		'''
			Constructor
			: param height : dimension of input feature space along spatial height
			: param width : dimension of input feature space along spatial width
			: param depth : dimension of input feature space along spatial depth
			: param extent : dimension of input feature space convoluted along spatial dimension
			: param alpha : learning rate constant hyperparameter
			: param stride : dimension of input feature space skipped between convolutions
			: param padding : dimension of padding added to periphery of spatial dimension
		'''
		Convolution.__init__(self, height, width, depth, extent, stride, padding)
		layer.Layer.__init__(self, self.dimensions['height'] * self.dimensions['width'] * self.dimensions['depth'], self.dimensions['rows'] * self.dimensions['columns'], alpha)
		if not hasattr(self, 'parameters'):
			self.parameters = dict()
		self.parameters['weights'] = numpy.ones((1, self.dimensions['extent'] * self.dimensions['extent'] * self.dimensions['depth']), dtype = float)
		self.parameters['biases'] = numpy.ones((1, 1), dtype = float)
		self.cleardeltas()

	def applylearningrate(self, alpha = None):
		'''
			Method to apply learning gradient descent optimization
			: param alpha : learning rate constant hyperparameter
		'''
		if alpha is None:
			alpha = 0.05 / self.dimensions['outputs'] # default set to 0.05 / output_units
 		self.units['modifier'].applylearningrate(alpha)

	def feedforward(self, inputvector): # ignores dropout
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['output'].append(numpy.empty((self.dimensions['outputs'], 1), dtype = float))
		for row in range(self.dimensions['rows']):
			for column in range(self.dimensions['columns']):
				self.history['output'][-1][row * self.dimensions['columns'] + column][0] = configure.functions['add'](configure.functions['dot'](self.parameters['weights'], self.convolute(row, column, self.history['input'][-1])), self.parameters['biases'])
		return self.history['output'][-1]

	def backpropagate(self, outputvector): # ignores dropout
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['output'].pop()
		backvector = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
		for row in range(self.dimensions['rows']):
			for column in range(self.dimensions['columns']):
				self.deltaparameters['weights'] = configure.functions['add'](self.deltaparameters['weights'], configure.functions['multiply'](outputvector[row * self.dimensions['columns'] + column][0], configure.functions['transpose'](self.convolute(row, column, self.history['input'][-1]))))
				self.deltaparameters['biases'] = configure.functions['add'](outputvector[row * self.dimensions['columns'] + column][0], self.deltaparameters['biases'])
				backvector = self.unconvolute(row, column, configure.functions['multiply'](outputvector[row * self.dimensions['columns'] + column][0], configure.functions['transpose'](self.parameters['weights'])), backvector)
		self.history['input'].pop()
		return backvector

class Pooling(Convolution):
	'''
		Base Class for Spatial Pooling Layers
	'''
	def __init__(self, height, width, depth, extent, stride = None):
		'''
			Constructor
			: param height : dimension of input feature space along spatial height
			: param width : dimension of input feature space along spatial width
			: param depth : dimension of input feature space along spatial depth
			: param extent : dimension of input feature space convoluted along spatial dimension
			: param stride : dimension of input feature space skipped between convolutions
		'''
		stride = stride if stride is not None else extent
		Convolution.__init__(self, height, width, depth, extent, stride, 0)
		self.dimensions['inputs'] = self.dimensions['height'] * self.dimensions['width'] * self.dimensions['depth']
		self.dimensions['outputs'] = self.dimensions['rows'] * self.dimensions['columns']
		self.history['input'] = list()
		self.history['output'] = list()
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
		self.history['output'].append(numpy.empty((self.dimensions['outputs'], 1), dtype = float))
		for row in range(self.dimensions['rows']):
			for column in range(self.dimensions['columns']):
				self.history['output'][-1][row * self.dimensions['columns'] + column][0] = self.functions['function'](self.convolute(row, column, self.history['input'][-1]))
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		backvector = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
		for row in range(self.dimensions['rows']):
			for column in range(self.dimensions['columns']):
				backvector = self.unconvolute(row, column, self.functions['derivative'](self.convolute(row, column, self.history['input'][-1]), self.history['output'][-1][row * self.dimensions['columns'] + column][0], outputvector[row * self.dimensions['columns'] + column][0]), backvector)
		self.history['output'].pop()
		self.history['input'].pop()
		return backvector

class MaxPooling(Pooling):
	'''
		Maximum Pooling Layer
		Mathematically, f(x) = [max(conv(x))]
	'''
	def __init__(self, height, width, depth, extent, stride = None):
		'''
			Constructor
			: param height : dimension of input feature space along spatial height
			: param width : dimension of input feature space along spatial width
			: param depth : dimension of input feature space along spatial depth
			: param extent : dimension of input feature space convoluted along spatial dimension
			: param stride : dimension of input feature space skipped between convolutions
		'''
		Pooling.__init__(self, height, width, depth, extent, stride)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['amax']
		self.functions['derivative'] = configure.functions['vectorize'](lambda x, y, z: z if x == y else 0.0)

class MinPooling(Pooling):
	'''
		Minimum Pooling Layer
		Mathematically, f(x) = [min(conv(x))]
	'''
	def __init__(self, height, width, depth, extent, stride = None):
		'''
			Constructor
			: param height : dimension of input feature space along spatial height
			: param width : dimension of input feature space along spatial width
			: param depth : dimension of input feature space along spatial depth
			: param extent : dimension of input feature space convoluted along spatial dimension
			: param stride : dimension of input feature space skipped between convolutions
		'''
		Pooling.__init__(self, height, width, depth, extent, stride)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['amin']
		self.functions['derivative'] = configure.functions['vectorize'](lambda x, y, z: z if x == y else 0.0)

class AveragePooling(Pooling):
	'''
		Average Pooling Layer
		Mathematically, f(x) = [avg(conv(x))]
	'''
	def __init__(self, height, width, depth, extent, stride = None):
		'''
			Constructor
			: param height : dimension of input feature space along spatial height
			: param width : dimension of input feature space along spatial width
			: param depth : dimension of input feature space along spatial depth
			: param extent : dimension of input feature space convoluted along spatial dimension
			: param stride : dimension of input feature space skipped between convolutions
		'''
		Pooling.__init__(self, height, width, depth, extent, stride)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = configure.functions['mean']
		self.functions['derivative'] = configure.functions['vectorize'](lambda x, y, z: z / (self.dimensions['extent'] * self.dimensions['extent'] * self.dimensions['depth']))
