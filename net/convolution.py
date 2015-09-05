import numpy, layer

class Convolution:

	height = None
	width = None
	depth = None

	rows = None
	columns = None

	stride = None
	extent = None
	padding = None

	def __init__(self, height, width, depth, extent, stride = None, padding = None):
		self.height = height
		self.width = width
		self.depth = depth
		self.extent = extent
		self.stride = stride if stride is not None else 1
		self.padding = padding if padding is not None else 0
		self.rows = (self.height - self.extent + 2 * self.padding) / self.stride + 1
		self.columns = (self.width - self.extent + 2 * self.padding) / self.stride + 1

	def convolute(self, outrow, outcolumn, invector):
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
		for depthextent in range(self.depth):
			for rowextent in range(self.extent):
				inrow = - self.padding + outrow * self.stride + rowextent
				for columnextent in range(self.extent):
					incolumn = - self.padding + outcolumn * self.stride + columnextent
					if inrow in range(self.height) and incolumn in range(self.width):
						backvector[(depthextent * self.height + inrow) * self.width + incolumn][0] += outvector[(depthextent * self.extent + rowextent) * self.extent + columnextent][0]
		return backvector

class Convolutional(layer.Layer, Convolution):

	def __init__(self, height, width, depth, extent, alpha = None, eta = None, stride = None, padding = None):
		Convolution.__init__(self, height, width, depth, extent, stride, padding)
		layer.Layer.__init__(self, self.height * self.width * self.depth, alpha, eta)
		self.outputs = self.rows * self.columns
		self.weights = numpy.ones((1, self.extent * self.extent * self.depth), dtype = float)
		self.biases = numpy.ones((1, 1), dtype = float)
		self.cleardeltas()

	def feedforward(self, inputvector): # ignores dropout
		self.previousinput = inputvector
		self.previousoutput = numpy.empty((self.outputs, 1), dtype = float)
		for row in range(self.rows):
			for column in range(self.columns):
				self.previousoutput[row * self.columns + column][0] = numpy.add(numpy.dot(self.weights, self.convolute(row, column, self.previousinput)), self.biases)
		return self.previousoutput

	def backpropagate(self, outputvector): # ignores dropout
		backvector = numpy.zeros((self.inputs, 1), dtype = float)
		for row in range(self.rows):
			for column in range(self.columns):
				self.deltaweights = numpy.add(self.deltaweights, numpy.multiply(outputvector[row * self.columns + column][0], self.convolute(row, column, self.previousinput).transpose()))
				self.deltabiases = numpy.add(outputvector[row * self.columns + column][0], self.deltabiases)
				backvector = self.unconvolute(row, column, numpy.multiply(outputvector[row * self.columns + column][0], self.weights.transpose()), backvector)
		return backvector

class Pooling(Convolution):

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	function = None
	derivative = None

	def __init__(self, height, width, depth, extent, stride = None):
		stride = stride if stride is not None else extent
		Convolution.__init__(self, height, width, depth, extent, stride, 0)
		self.inputs = self.height * self.width * self.depth
		self.outputs = self.rows * self.columns

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = numpy.empty((self.outputs, 1), dtype = float)
		for row in range(self.rows):
			for column in range(self.columns):
				self.previousoutput[row * self.columns + column][0] = self.function(self.convolute(row, column, self.previousinput))
		return self.previousoutput

	def backpropagate(self, outputvector):
		backvector = numpy.zeros((self.inputs, 1), dtype = float)
		for row in range(self.rows):
			for column in range(self.columns):
				backvector = self.unconvolute(row, column, self.derivative(self.convolute(row, column, self.previousinput), self.previousoutput[row * self.columns + column][0], outputvector[row * self.columns + column][0]), backvector)
		return backvector

class MaxPooling(Pooling):

	def __init__(self, height, width, depth, extent, stride = None):
		Pooling.__init__(self, height, width, depth, extent, stride)
		self.function = numpy.amax
		self.derivative = numpy.vectorize(lambda x, y, z: z if x == y else 0.0)

class MinPooling(Pooling):

	def __init__(self, height, width, depth, extent, stride = None):
		Pooling.__init__(self, height, width, depth, extent, stride)
		self.function = numpy.amin
		self.derivative = numpy.vectorize(lambda x, y, z: z if x == y else 0.0)

class AveragePooling(Pooling):

	def __init__(self, height, width, depth, extent, stride = None):
		Pooling.__init__(self, height, width, depth, extent, stride)
		self.function = numpy.mean
		self.derivative = numpy.vectorize(lambda x, y, z: z / (self.extent * self.extent * self.depth))
