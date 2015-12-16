'''
	Module containing Self Organising Feature Maps.
	Classes embody Parametric Layers,
	used to learn low dimensional representations of data.
'''
import math, numpy
from . import configure, layer, error

class SelfOrganising(layer.Layer):
	'''
		Base Class for Self Organising Feature Maps
		Mathematically, f(x)(i) = 1.0 if i = argmin(r(i))
								= 0.0 otherwise
	'''
	exponentialneighbourhood = configure.functions['vectorize'](lambda x, y: math.exp(x - y))
	inverseneighbourhood = configure.functions['vectorize'](lambda x, y: 1.0 / (1.0 + y - x))
	knockerneighbourhood = configure.functions['vectorize'](lambda x, y: 1.0 if x == y else 0.0)

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
		self.history['output'].append(numpy.zeros((self.dimensions['outputs'], 1), dtype = float))
		for i in range(self.dimensions['outputs']):
			self.history['radius'][-1][i][0] = self.functions['function'](self.history['input'][-1], self.parameters['weights'][i].reshape((self.dimensions['inputs'], 1)))
		self.history['output'][-1][configure.functions['argmin'](self.history['radius'][-1])][0] = 1.0
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
		index = configure.functions['argmin'](self.history['radius'][-1])
		centrevector = self.parameters['weights'][index].reshape((self.dimensions['inputs'], 1))
		self.deltaparameters['weights'][index] = configure.functions['add'](self.deltaparameters['weights'][index], configure.functions['multiply'](outputvector[index][0], configure.functions['transpose'](self.functions['weightsderivative'](self.history['radius'][-1][index][0], self.history['input'][-1], centrevector))))
		return configure.functions['multiply'](outputvector[index][0], self.functions['functionderivative'](self.history['radius'].pop()[index][0], self.history['input'].pop(), centrevector))

	def pretrain(self, trainingset, threshold = 0.0001, batch = 1, iterations = 10, neighbourhood = None):
		'''
			Method to pretrain parameters using Competitive Learning
			: param trainingset : unsupervised training set
			: param threshold : distance from centre vector threshold for termination
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param neighbourhood : competitive learning update neighbour function
			: returns : elementwise reconstruction error on termination
		'''
		if neighbourhood is None:
			neighbourhood = SelfOrganising.exponentialneighbourhood
		self.trainingsetup()
		for i in range(iterations):
			for j in range(len(trainingset)):
				if j % batch == 0:
					self.updateweights()
				self.feedforward(trainingset[j])
				closest = self.history['radius'][-1][configure.functions['argmin'](self.history['output'][-1])][0]
				for k in range(self.dimensions['outputs']):
					factor = neighbourhood(closest, self.history['radius'][-1][k][0])
					centrevector = self.parameters['weights'][k].reshape((self.dimensions['inputs'], 1))
					self.deltaparameters['weights'][k] = configure.functions['add'](self.deltaparameters['weights'][k], configure.functions['multiply'](factor, configure.functions['transpose'](configure.functions['subtract'](centrevector, trainingset[j]))))
				self.history['output'].pop()
				self.history['radius'].pop()
				self.history['input'].pop()
			maximumdistance = float('-inf')
			for vector in trainingset:
				self.feedforward(vector)
				closest = self.history['radius'][-1][configure.functions['argmin'](self.history['output'][-1])][0]
				if closest > maximumdistance:
					maximumdistance = closest
				self.history['output'].pop()
				self.history['radius'].pop()
				self.history['input'].pop()
			if maximumdistance < threshold:
				break
		return maximumdistance

class ManhattanSO(SelfOrganising):
	'''
		Manhattan Distance Self Organising Map
		Mathematically, r(i) = sum_over_j(|x(j) - w(i)(j)|)
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		SelfOrganising.__init__(self, inputs, outputs, alpha)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = lambda inputvector, centrevector: configure.functions['sum'](configure.functions['fabs'](configure.functions['subtract'](inputvector, centrevector)))
		self.functions['functionderivative'] = lambda inputradius, inputvector, centrevector: configure.functions['sign'](configure.functions['subtract'](inputvector, centrevector))
		self.functions['weightsderivative'] = lambda inputradius, inputvector, centrevector: configure.functions['sign'](configure.functions['subtract'](centrevector, inputvector))

class EuclideanSquaredSO(SelfOrganising):
	'''
		Euclidean Distance Self Organising Map
		Mathematically, r(i) = sum_over_j((x(j) - w(i)(j)) ^ 2) ^ 0.5
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		SelfOrganising.__init__(self, inputs, outputs, alpha)

	def __finit__(self):
		'''
			Internal Method used to initialize function attributes
		'''
		if not hasattr(self, 'functions'):
			self.functions = dict()
		self.functions['function'] = lambda inputvector, centrevector: configure.functions['sqrt'](configure.functions['sum'](configure.functions['square'](configure.functions['subtract'](inputvector, centrevector))))
		self.functions['functionderivative'] = lambda inputradius, inputvector, centrevector: configure.functions['divide'](configure.functions['subtract'](inputvector, centrevector), inputradius)
		self.functions['weightsderivative'] = lambda inputradius, inputvector, centrevector: configure.functions['divide'](configure.functions['subtract'](centrevector, inputvector), inputradius)
