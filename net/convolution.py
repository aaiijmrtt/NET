'''
	Module containing Convolution Layers.
	Classes embody Spatial Convolutions,
	used to exploit spatial coherence.
'''
import numpy
from . import layer

class Convolution:
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
		self.height = height
		self.width = width
		self.depth = depth
		self.extent = extent
		self.stride = stride if stride is not None else 1
		self.padding = padding if padding is not None else 0
		self.rows = (self.height - self.extent + 2 * self.padding) / self.stride + 1
		self.columns = (self.width - self.extent + 2 * self.padding) / self.stride + 1

	def convolute(self, outrow, outcolumn, invector):
		'''
			Method to perform spatial convolution during feedforward
			: param outrow : index in output feature space along spatial height
			: param outcolumn : index in output feature space along spatial width
			: param invector : vector in input feature space
			: returns : convoluted vector mapped to output feature space
		'''
		vector = numpy.empty((self.extent * self.extent * self.depth, 1), dtype = float)
		for depthextent in range(self.depth):
			for rowextent in range(self.extent):
				inrow = - self.padding + outrow * self.stride + rowextent
				for columnextent in range(self.extent):
					incolumn = - self.padding + outcolumn * self.stride + columnextent
					if inrow in range(self.height) and incolumn in range(self.width):
						vector[(depthextent * self.extent + rowextent) * self.extent + columnextent][0] = invector[(depthextent * self.height + inrow) * self.width + incolumn][0]
					else:
						vector[(depthextent * self.extent + rowextent) * self.extent + columnextent][0] = 0.0
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
		for depthextent in range(self.depth):
			for rowextent in range(self.extent):
				inrow = - self.padding + outrow * self.stride + rowextent
				for columnextent in range(self.extent):
					incolumn = - self.padding + outcolumn * self.stride + columnextent
					if inrow in range(self.height) and incolumn in range(self.width):
						backvector[(depthextent * self.height + inrow) * self.width + incolumn][0] += outvector[(depthextent * self.extent + rowextent) * self.extent + columnextent][0]
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
		layer.Layer.__init__(self, self.height * self.width * self.depth, alpha)
		self.outputs = self.rows * self.columns
		self.parameters = dict()
		self.parameters['weights'] = numpy.ones((1, self.extent * self.extent * self.depth), dtype = float)
		self.parameters['biases'] = numpy.ones((1, 1), dtype = float)
		self.cleardeltas()

	def feedforward(self, inputvector): # ignores dropout
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = inputvector
		self.previousoutput = numpy.empty((self.outputs, 1), dtype = float)
		for row in range(self.rows):
			for column in range(self.columns):
				self.previousoutput[row * self.columns + column][0] = numpy.add(numpy.dot(self.parameters['weights'], self.convolute(row, column, self.previousinput)), self.parameters['biases'])
		return self.previousoutput

	def backpropagate(self, outputvector): # ignores dropout
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		backvector = numpy.zeros((self.inputs, 1), dtype = float)
		for row in range(self.rows):
			for column in range(self.columns):
				self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.multiply(outputvector[row * self.columns + column][0], numpy.transpose(self.convolute(row, column, self.previousinput))))
				self.deltaparameters['biases'] = numpy.add(outputvector[row * self.columns + column][0], self.deltaparameters['biases'])
				backvector = self.unconvolute(row, column, numpy.multiply(outputvector[row * self.columns + column][0], numpy.transpose(self.parameters['weights'])), backvector)
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
		self.inputs = self.height * self.width * self.depth
		self.outputs = self.rows * self.columns
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
		self.previousoutput = numpy.empty((self.outputs, 1), dtype = float)
		for row in range(self.rows):
			for column in range(self.columns):
				self.previousoutput[row * self.columns + column][0] = self.function(self.convolute(row, column, self.previousinput))
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		backvector = numpy.zeros((self.inputs, 1), dtype = float)
		for row in range(self.rows):
			for column in range(self.columns):
				backvector = self.unconvolute(row, column, self.derivative(self.convolute(row, column, self.previousinput), self.previousoutput[row * self.columns + column][0], outputvector[row * self.columns + column][0]), backvector)
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
		self.function = numpy.amax
		self.derivative = numpy.vectorize(lambda x, y, z: z if x == y else 0.0)

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
		self.function = numpy.amin
		self.derivative = numpy.vectorize(lambda x, y, z: z if x == y else 0.0)

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
		self.function = numpy.mean
		self.derivative = numpy.vectorize(lambda x, y, z: z / (self.extent * self.extent * self.depth))
