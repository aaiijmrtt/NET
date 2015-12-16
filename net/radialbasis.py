'''
	Module containing Radial Basis Functions.
	Classes embody Parametric Radial Basis Layers.
'''
import math, numpy
from . import configure, layer, error

class RadialBasis(layer.Layer):
	'''
		Base Class for Radial Basis Functions
		Mathematically, r(i) = (sum_over_j((x(j) - p1(i)(j)) ^ 2)) ^ 0.5
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		layer.Layer.__init__(self, inputs, outputs, alpha)
		self.history['radius'] = list()
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], self.dimensions['inputs']))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.dimensions['inputs']), (self.dimensions['outputs'], 1))
		self.__finit__()
		self.cleardeltas()

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = None
		self.functions['functionderivative'] = None
		self.functions['weightsderivative'] = None
		self.functions['biasesderivative'] = None

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		self.history['input'].append(inputvector)
		self.history['radius'].append(numpy.empty((self.dimensions['outputs'], 1), dtype = float))
		self.history['output'].append(numpy.empty((self.dimensions['outputs'], 1), dtype = float))
		for i in range(self.dimensions['outputs']):
			centrevector = self.parameters['weights'][i].reshape((self.dimensions['inputs'], 1))
			self.history['radius'][-1][i][0] = configure.functions['sum'](configure.functions['square'](configure.functions['subtract'](self.history['input'][-1], centrevector))) ** 0.5
			self.history['output'][-1][i][0] = self.functions['function'](self.history['radius'][-1][i][0], self.parameters['biases'][i][0])
		return self.history['output'][-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		deltainputs = numpy.zeros((self.dimensions['inputs'], 1), dtype = float)
		for i in range(self.dimensions['outputs']):
			centrevector = self.parameters['weights'][i].reshape((self.dimensions['inputs'], 1))
			deltainputs = configure.functions['add'](deltainputs, configure.functions['multiply'](outputvector[i][0], self.functions['functionderivative'](self.history['radius'][-1][i][0], self.history['input'][-1], centrevector, self.parameters['biases'][i][0], self.history['output'][-1][i][0])))
			self.deltaparameters['weights'][i] = configure.functions['add'](self.deltaparameters['weights'][i], configure.functions['multiply'](outputvector[i][0], configure.functions['transpose'](self.functions['weightsderivative'](self.history['radius'][-1][i][0], self.history['input'][-1], centrevector, self.parameters['biases'][i][0], self.history['output'][-1][i][0]))))
			self.deltaparameters['biases'][i][0] += outputvector[i][0] * self.functions['biasesderivative'](self.history['radius'][-1][i][0], self.history['input'][-1], centrevector, self.parameters['biases'][i][0], self.history['output'][-1][i][0])
		self.history['output'].pop()
		self.history['radius'].pop()
		self.history['input'].pop()
		return deltainputs

	def pretrain(self, trainingset, threshold = 0.0001, iterations = 10, criterion = None):
		'''
			Method to pretrain parameters using K Means Clustering
			: param trainingset : unsupervised training set
			: param threshold : distance from cluster centre threshold for termination
			: param iterations : iteration threshold for termination
			: param criterion : criterion used to quantify distance from cluster centre
			: returns : maximum distance from cluster centre on termination
		'''
		if criterion is None:
			criterion = error.MeanSquared(self.dimensions['inputs'])
		for iterations in range(iterations):
			clusters = [[self.parameters['weights'][i].reshape((self.dimensions['inputs'], 1))] for i in range(self.dimensions['outputs'])]
			for point in trainingset:
				bestdistance = float('inf')
				bestindex = -1
				for i in range(len(clusters)):
					distance = configure.functions['sum'](criterion.compute(point, clusters[i][0]))
					if distance < bestdistance:
						bestdistance = distance
						bestindex = i
				clusters[bestindex].append(point)
			for cluster in clusters:
				if len(cluster) > 1:
					cluster[0] = configure.functions['mean'](cluster[1:], axis = 0)
			maximumdistance = float('-inf')
			for cluster in clusters:
				if len(cluster) > 1:
					for point in cluster[1: ]:
						distance = configure.functions['sum'](criterion.compute(point, cluster[0]))
						if distance > maximumdistance:
							maximumdistance = distance
			if maximumdistance < threshold:
				break
		for i in range(len(self.parameters['weights'])):
			self.parameters['weights'][i] = configure.functions['transpose'](clusters[i][0])
		return maximumdistance

class GaussianRB(RadialBasis):
	'''
		Gaussian Radial Basis Function
		Mathematically, f(x)(i) = exp(- r(i) ^ 2 / p2 ^ 2)
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		RadialBasis.__init__(self, inputs, outputs, alpha)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = lambda inputradius, coefficient: configure.functions['exp'](-inputradius / coefficient ** 2)
		self.functions['functionderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply'](2.0 * output / (coefficient ** 2), configure.functions['subtract'](centrevector, inputvector))
		self.functions['weightsderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply'](2.0 * output / (coefficient ** 2), configure.functions['subtract'](inputvector, centrevector))
		self.functions['biasesderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: 2.0 * output * configure.functions['sum'](configure.functions['square'](configure.functions['subtract'](inputvector, centrevector))) / (coefficient ** 3)

class MultiQuadraticRB(RadialBasis):
	'''
		Multi Quadratic Radial Basis Function
		Mathematically, f(x)(i) = (r(i) ^ 2 + p2 ^ 2) ^ p3
	'''
	def __init__(self, inputs, outputs, alpha = None, beta = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
			: param beta : p3, as given in its mathematical expression
		'''
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.metaparameters['beta'] = beta if beta is not None else 0.5

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = lambda inputradius, coefficient: (inputradius ** 2 + coefficient ** 2) ** self.metaparameters['beta']
		self.functions['functionderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply'](self.metaparameters['beta'] * (inputradius ** 2 + coefficient ** 2) ** (self.metaparameters['beta'] - 1.0) * 2.0, configure.functions['subtract'](inputvector, centrevector))
		self.functions['weightsderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply'](self.metaparameters['beta'] * (inputradius ** 2 + coefficient ** 2) ** (self.metaparameters['beta'] - 1.0) * 2.0, configure.functions['subtract'](centrevector, inputvector))
		self.functions['biasesderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: self.metaparameters['beta'] * (inputradius ** 2 + coefficient ** 2) ** (self.metaparameters['beta'] - 1.0) * 2.0 * coefficient

class InverseMultiQuadraticRB(RadialBasis):
	'''
		Inverse Multi Quadratic Radial Basis Function
		Mathematically, f(x)(i) = (r(i) ^ 2 + p2 ^ 2) ^ (-p3)
	'''
	def __init__(self, inputs, outputs, alpha = None, beta = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
			: param beta : p3, as given in its mathematical expression
		'''
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.metaparameters['beta'] = beta if beta is not None else 0.5

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = lambda inputradius, coefficient: (inputradius ** 2 + coefficient ** 2) ** (-self.metaparameters['beta'])
		self.functions['functionderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply'](-self.metaparameters['beta'] * (inputradius ** 2 + coefficient ** 2) ** (-self.metaparameters['beta'] - 1.0) * 2.0, configure.functions['subtract'](inputvector, centrevector))
		self.functions['weightsderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply'](-self.metaparameters['beta'] * (inputradius ** 2 + coefficient ** 2) ** (-self.metaparameters['beta'] - 1.0) * 2.0, configure.functions['subtract'](centrevector, inputvector))
		self.functions['biasesderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: -self.metaparameters['beta'] * (inputradius ** 2 + coefficient ** 2) ** (-self.metaparameters['beta'] - 1.0) * 2.0 * coefficient

class ThinPlateSplineRB(RadialBasis):
	'''
		Thin Plate Spline Radial Basis Function
		Mathematically, f(x)(i) = r(i) ^ 2 * log(r(i))
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		RadialBasis.__init__(self, inputs, outputs, alpha)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = lambda inputradius, coefficient: inputradius ** 2 * configure.functions['log'](inputradius)
		self.functions['functionderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply']((2.0 * configure.functions['log'](inputradius) + 1.0), configure.functions['subtract'](inputvector, centrevector))
		self.functions['weightsderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply']((2.0 * configure.functions['log'](inputradius) + 1.0), configure.functions['subtract'](centrevector, inputvector))
		self.functions['biasesderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0

class CubicRB(RadialBasis):
	'''
		Cubic Radial Basis Function
		Mathematically, f(x)(i) = r(i) ^ 3
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		RadialBasis.__init__(self, inputs, outputs, alpha)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = lambda inputradius, coefficient: inputradius ** 3
		self.functions['functionderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply'](3.0 * inputradius, configure.functions['subtract'](inputvector, centrevector))
		self.functions['weightsderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['multiply'](3.0 * inputradius, configure.functions['subtract'](centrevector, inputvector))
		self.functions['biasesderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0

class LinearRB(RadialBasis):
	'''
		Linear Radial Basis Function
		Mathematically, f(x)(i) = r(i)
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		RadialBasis.__init__(self, inputs, outputs, alpha)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = lambda inputradius, coefficient: inputradius
		self.functions['functionderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['divide'](configure.functions['subtract'](inputvector, centrevector), output)
		self.functions['weightsderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: configure.functions['divide'](configure.functions['subtract'](centrevector, inputvector), output)
		self.functions['biasesderivative'] = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0
