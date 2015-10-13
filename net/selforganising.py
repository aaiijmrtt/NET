'''
	Module containing Self Organising Feature Maps.
	Classes embody Parametric Layers,
	used to learn low dimensional representations of data.
'''
import numpy
from . import layer, error

class SelfOrganising(layer.Layer):
	'''
		Base Class for Self Organising Feature Maps
		Mathematically, f(x)(i) = 1.0 if i = argmin(r(i))
								= 0.0 otherwise
	'''
	exponentialneighbourhood = numpy.vectorize(lambda x, y: numpy.exp(x - y))
	inverseneighbourhood = numpy.vectorize(lambda x, y: 1.0 / (1.0 + y - x))
	knockerneighbourhood = numpy.vectorize(lambda x, y: 1.0 if x == y else 0.0)

	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		layer.Layer.__init__(self, inputs, outputs, alpha)
		self.parameters = dict()
		self.deltaparameters = dict()
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
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
		self.previousinput = inputvector
		self.previousradius = numpy.empty((self.outputs, 1), dtype = float)
		self.previousoutput = numpy.zeros((self.outputs, 1), dtype = float)
		for i in range(self.outputs):
			self.previousradius[i][0] = self.function(self.previousinput, self.parameters['weights'][i].reshape((self.inputs, 1)))
		self.previousoutput[numpy.argmin(self.previousradius)][0] = 1.0
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		index = numpy.argmin(self.previousradius)
		centrevector = self.parameters['weights'][index].reshape((self.inputs, 1))
		self.deltaparameters['weights'][index] = numpy.add(self.deltaparameters['weights'][index], numpy.multiply(outputvector[index][0], numpy.transpose(self.weightsderivative(self.previousradius[index][0], self.previousinput, centrevector))))
		return numpy.multiply(outputvector[index][0], self.functionderivative(self.previousradius[index][0], self.previousinput, centrevector))

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
				closest = self.previousradius[numpy.argmin(self.previousoutput)][0]
				for k in range(self.outputs):
					factor = neighbourhood(closest, self.previousradius[k][0])
					centrevector = self.parameters['weights'][k].reshape((self.inputs, 1))
					self.deltaparameters['weights'][k] = numpy.add(self.deltaparameters['weights'][k], numpy.multiply(factor, numpy.transpose(numpy.subtract(centrevector, trainingset[j]))))
			maximumdistance = float('-inf')
			for vector in trainingset:
				self.feedforward(vector)
				closest = self.previousradius[numpy.argmin(self.previousoutput)][0]
				if closest > maximumdistance:
					maximumdistance = closest
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
		self.function = lambda inputvector, centrevector: numpy.sum(numpy.abs(numpy.subtract(inputvector, centrevector)))
		self.functionderivative = lambda inputradius, inputvector, centrevector: numpy.sign(numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector: numpy.sign(numpy.subtract(centrevector, inputvector))

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
		self.function = lambda inputvector, centrevector: numpy.sqrt(numpy.sum(numpy.square(numpy.subtract(inputvector, centrevector))))
		self.functionderivative = lambda inputradius, inputvector, centrevector: numpy.divide(numpy.subtract(inputvector, centrevector), inputradius)
		self.weightsderivative = lambda inputradius, inputvector, centrevector: numpy.divide(numpy.subtract(centrevector, inputvector), inputradius)
