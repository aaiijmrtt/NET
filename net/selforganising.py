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
		self.previousradius = list()
		self.parameters = dict()
		self.deltaparameters = dict()
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / math.sqrt(self.inputs), (self.outputs, self.inputs))
		self.function = None
		self.functionderivative = None
		self.weightsderivative = None
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.previousradius.append(numpy.empty((self.outputs, 1), dtype = float))
		self.previousoutput.append(numpy.zeros((self.outputs, 1), dtype = float))
		for i in range(self.outputs):
			self.previousradius[-1][i][0] = self.function(self.previousinput[-1], self.parameters['weights'][i].reshape((self.inputs, 1)))
		self.previousoutput[-1][configure.functions['argmin'](self.previousradius[-1])][0] = 1.0
		return self.previousoutput[-1]

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		index = configure.functions['argmin'](self.previousradius[-1])
		centrevector = self.parameters['weights'][index].reshape((self.inputs, 1))
		self.deltaparameters['weights'][index] = configure.functions['add'](self.deltaparameters['weights'][index], configure.functions['multiply'](outputvector[index][0], configure.functions['transpose'](self.weightsderivative(self.previousradius[-1][index][0], self.previousinput[-1], centrevector))))
		return configure.functions['multiply'](outputvector[index][0], self.functionderivative(self.previousradius.pop()[index][0], self.previousinput.pop(), centrevector))

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
				closest = self.previousradius[-1][configure.functions['argmin'](self.previousoutput[-1])][0]
				for k in range(self.outputs):
					factor = neighbourhood(closest, self.previousradius[-1][k][0])
					centrevector = self.parameters['weights'][k].reshape((self.inputs, 1))
					self.deltaparameters['weights'][k] = configure.functions['add'](self.deltaparameters['weights'][k], configure.functions['multiply'](factor, configure.functions['transpose'](configure.functions['subtract'](centrevector, trainingset[j]))))
				self.previousoutput.pop()
				self.previousradius.pop()
				self.previousinput.pop()
			maximumdistance = float('-inf')
			for vector in trainingset:
				self.feedforward(vector)
				closest = self.previousradius[-1][configure.functions['argmin'](self.previousoutput[-1])][0]
				if closest > maximumdistance:
					maximumdistance = closest
				self.previousoutput.pop()
				self.previousradius.pop()
				self.previousinput.pop()
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
		self.function = lambda inputvector, centrevector: configure.functions['sum'](configure.functions['fabs'](configure.functions['subtract'](inputvector, centrevector)))
		self.functionderivative = lambda inputradius, inputvector, centrevector: configure.functions['sign'](configure.functions['subtract'](inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector: configure.functions['sign'](configure.functions['subtract'](centrevector, inputvector))

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
		self.function = lambda inputvector, centrevector: configure.functions['sqrt'](configure.functions['sum'](configure.functions['square'](configure.functions['subtract'](inputvector, centrevector))))
		self.functionderivative = lambda inputradius, inputvector, centrevector: configure.functions['divide'](configure.functions['subtract'](inputvector, centrevector), inputradius)
		self.weightsderivative = lambda inputradius, inputvector, centrevector: configure.functions['divide'](configure.functions['subtract'](centrevector, inputvector), inputradius)
